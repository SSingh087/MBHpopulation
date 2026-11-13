"""
flows.py -- conditional autoregressive rational-quadratic-spline flow, written from scratch in PyTorch
(no zuko / nflows / sbi).  Spline math follows Durkan et al. 2019 (Neural Spline Flows) with linear tails.

ConditionalFlow(d, ctx_dim): models p(x | c) for x in R^d, c in R^ctx_dim.
  * context encoder: MLP(ctx_dim -> h -> h)
  * n_layers autoregressive spline layers, dimension order reversed every layer; in each layer the spline
    parameters for dimension i come from a small MLP of [encoded context, x_<i]  (d is tiny -> one MLP per dim)
  * base distribution: standard normal
  log_prob(x, c) = log N(z) + sum log|dz/dx|     sample(c): invert layer by layer (sequential over dims)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_W, MIN_H, MIN_D = 1e-3, 1e-3, 1e-3


def _rqs(x, uw, uh, ud, inverse, left, right, bottom, top):
    """Rational-quadratic spline on [left,right]->[bottom,top]. x:(n,), uw/uh:(n,K), ud:(n,K+1). Returns y, logabsdet."""
    K = uw.shape[-1]
    widths = F.softmax(uw, -1); widths = MIN_W + (1 - MIN_W * K) * widths
    cumw = F.pad(torch.cumsum(widths, -1), (1, 0), value=0.0)
    cumw = (right - left) * cumw + left
    cumw = torch.cat([cumw[..., :1] * 0 + left, cumw[..., 1:-1], cumw[..., -1:] * 0 + right], -1)
    widths = cumw[..., 1:] - cumw[..., :-1]
    derivs = MIN_D + F.softplus(ud)
    heights = F.softmax(uh, -1); heights = MIN_H + (1 - MIN_H * K) * heights
    cumh = F.pad(torch.cumsum(heights, -1), (1, 0), value=0.0)
    cumh = (top - bottom) * cumh + bottom
    cumh = torch.cat([cumh[..., :1] * 0 + bottom, cumh[..., 1:-1], cumh[..., -1:] * 0 + top], -1)
    heights = cumh[..., 1:] - cumh[..., :-1]

    loc = (cumh if inverse else cumw).clone()
    loc[..., -1] = loc[..., -1] + 1e-6
    idx = (torch.searchsorted(loc.contiguous(), x[..., None].contiguous()) - 1).clamp(0, K - 1)

    g = lambda t: t.gather(-1, idx)[..., 0]
    in_cw, in_w = g(cumw), g(widths)
    in_ch, in_h = g(cumh), g(heights)
    delta = heights / widths
    in_delta = g(delta)
    in_d, in_d1 = g(derivs), derivs[..., 1:].gather(-1, idx)[..., 0]

    if inverse:
        t1 = (x - in_ch) * (in_d + in_d1 - 2 * in_delta)
        a = t1 + in_h * (in_delta - in_d)
        b = in_h * in_d - t1
        c = -in_delta * (x - in_ch)
        disc = (b ** 2 - 4 * a * c).clamp_min(0)
        root = (2 * c) / (-b - torch.sqrt(disc))
        y = root * in_w + in_cw
        tt = root * (1 - root)
        den = in_delta + (in_d + in_d1 - 2 * in_delta) * tt
        dnum = in_delta ** 2 * (in_d1 * root ** 2 + 2 * in_delta * tt + in_d * (1 - root) ** 2)
        return y, -(torch.log(dnum) - 2 * torch.log(den))
    theta = (x - in_cw) / in_w
    tt = theta * (1 - theta)
    num = in_h * (in_delta * theta ** 2 + in_d * tt)
    den = in_delta + (in_d + in_d1 - 2 * in_delta) * tt
    y = in_ch + num / den
    dnum = in_delta ** 2 * (in_d1 * theta ** 2 + 2 * in_delta * tt + in_d * (1 - theta) ** 2)
    return y, torch.log(dnum) - 2 * torch.log(den)


def spline_1d(x, params, K, tail, inverse=False):
    """x:(n,), params:(n,3K-1): K widths, K heights, K-1 interior derivatives. Identity outside [-tail,tail]."""
    uw, uh, ud = params[:, :K], params[:, K:2 * K], params[:, 2 * K:]
    const = math.log(math.exp(1 - MIN_D) - 1)
    ud = F.pad(ud, (1, 1), value=const)
    inside = (x >= -tail) & (x <= tail)
    xc = x.clamp(-tail, tail)                       # evaluate everywhere (no masking / host sync), select afterwards
    yi, li = _rqs(xc, uw, uh, ud, inverse, -tail, tail, -tail, tail)
    return torch.where(inside, yi, x), torch.where(inside, li, torch.zeros_like(li))


def _mlp(i, o, h, nl=2):
    layers, d = [], i
    for _ in range(nl):
        layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ARSplineLayer(nn.Module):
    def __init__(self, d, ctx_h, hidden, K, tail, reverse):
        super().__init__()
        self.d, self.K, self.tail, self.reverse = d, K, tail, reverse
        self.nets = nn.ModuleList([_mlp(ctx_h + i, 3 * K - 1, hidden) for i in range(d)])
        for net in self.nets:                              # start near identity
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def _order(self, x):
        return x.flip(-1) if self.reverse else x

    def forward(self, x, h):                               # x -> z, parallel over dims
        x = self._order(x)
        zs, lds = [], torch.zeros(x.shape[0], device=x.device)
        for i in range(self.d):
            p = self.nets[i](torch.cat([h, x[:, :i]], -1))
            zi, li = spline_1d(x[:, i], p, self.K, self.tail)
            zs.append(zi); lds = lds + li
        return self._order(torch.stack(zs, -1)), lds

    def inverse(self, z, h):                               # z -> x, sequential over dims
        z = self._order(z)
        x = torch.zeros_like(z)
        for i in range(self.d):
            p = self.nets[i](torch.cat([h, x[:, :i]], -1))
            xi, _ = spline_1d(z[:, i], p, self.K, self.tail, inverse=True)
            x = torch.cat([x[:, :i], xi[:, None], x[:, i + 1:]], -1)
        return self._order(x)


class ConditionalFlow(nn.Module):
    def __init__(self, d, ctx_dim, n_layers=6, hidden=64, K=8, tail=5.0, ctx_hidden=64):
        super().__init__()
        self.d, self.ctx_dim = d, ctx_dim
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, ctx_hidden), nn.SiLU(), nn.Linear(ctx_hidden, ctx_hidden), nn.SiLU())
        self.layers = nn.ModuleList([ARSplineLayer(d, ctx_hidden, hidden, K, tail, reverse=(l % 2 == 1)) for l in range(n_layers)])

    def log_prob(self, x, c):
        h = self.ctx(c)
        ld = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, l = layer(x, h); ld = ld + l
        return -0.5 * (x ** 2).sum(-1) - 0.5 * self.d * math.log(2 * math.pi) + ld

    @torch.no_grad()
    def sample(self, c):
        h = self.ctx(c)
        z = torch.randn(c.shape[0], self.d, device=c.device)
        for layer in reversed(self.layers):
            z = layer.inverse(z, h)
        return z
