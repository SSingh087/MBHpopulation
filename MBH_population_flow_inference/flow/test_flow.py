#!/usr/bin/env python
"""Correctness tests for flows.py: spline invertibility, flow inverse consistency, normalisation, gradients."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows import ConditionalFlow, spline_1d
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# 1. spline forward/inverse + logdet vs autograd
K, n, tail = 8, 2000, 5.0
x = (torch.rand(n) * 14 - 7).requires_grad_(True)               # includes points outside the tails
p = torch.randn(n, 3 * K - 1)
y, ld = spline_1d(x, p, K, tail)
xr, ldi = spline_1d(y.detach(), p, K, tail, inverse=True)
g = torch.autograd.grad(y.sum(), x)[0]
print("spline: max |x - inv(f(x))|     =", (x.detach() - xr).abs().max().item())
print("spline: max |logdet - log f'|   =", (ld - g.abs().log()).abs().max().item())
print("spline: max |fwd+inv logdet|    =", (ld + ldi).abs().max().item())

# 2. flow: mildly perturbed (non-identity) weights; inverse consistency, normalisation, gradients
for d in (2, 4):
    flow = ConditionalFlow(d, ctx_dim=6, n_layers=4, hidden=32, K=8)
    for prm in flow.parameters():
        prm.data += 0.1 * torch.randn_like(prm)
    c = torch.randn(1, 6)
    xs = flow.sample(c.expand(500, -1))
    h = flow.ctx(c.expand(500, -1)); w = xs
    for layer in flow.layers:
        w, _ = layer(w, h)
    print(f"flow d={d}: sample finite={torch.isfinite(xs).all().item()}  round-trip x->z is N(0,1): mean {w.mean().item():+.3f} std {w.std().item():.3f}")
    with torch.no_grad():
        if d == 2:                                          # direct grid integral
            g = torch.linspace(-9, 9, 1201); X, Y = torch.meshgrid(g, g, indexing="ij"); pts = torch.stack([X.ravel(), Y.ravel()], -1)
            lp = flow.log_prob(pts, c.expand(len(pts), -1)); dx = (g[1] - g[0]).item()
            print(f"flow d=2: grid integral p(x|c) dx = {(lp.exp().sum() * dx * dx).item():.4f}  (expect 1.0000)")
        else:                                               # MC with a wide Gaussian proposal
            N, s = 2_000_000, 3.0
            xg = torch.randn(N, d) * s
            logq = -0.5 * ((xg / s) ** 2).sum(-1) - d * torch.log(torch.tensor(s)) - 0.5 * d * torch.log(torch.tensor(2 * torch.pi))
            lp = flow.log_prob(xg, c.expand(N, -1))
            print(f"flow d=4: MC integral p(x|c) dx = {torch.exp(lp - logq).mean().item():.3f}  (expect 1.000, MC error ~1%)")
    loss = -flow.log_prob(xs.detach(), c.expand(500, -1)).mean(); loss.backward()
    print(f"flow d={d}: grads finite = {all(torch.isfinite(p.grad).all().item() for p in flow.parameters() if p.grad is not None)}")
