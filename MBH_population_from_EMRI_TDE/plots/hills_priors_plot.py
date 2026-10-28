"""How the beta prior and the Hills cut interact.
Detectability labels of tde_fallback do not depend on beta, so a different beta prior can be applied by
re-weighting each existing event with P(beta <= beta_max(M) | prior). NOT valid for cooling_envelope (beta changes its light curve)."""
import os; HERE = os.path.dirname(os.path.abspath(__file__))
import os, h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA='./data/astrophysical_data/100000'
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e4e3df"
C_UNI,C_FLC,C_SM="#2a78d6","#eb6834","#1baf7a"
Rsun=6.957e10; rS=2.953e5; BMAX_PRIOR=30.0
def bmax(lgM): return Rsun*(10**lgM)**(1/3)/(rS*10**lgM)           # r_p > r_S  (Stone & Metzger beta_max = Rt/2Rg)
def tail_frac(bm):                                                  # fraction of beta^-2 on [1,30] below bm
    """Fraction of the prior above bmax."""
    b=np.clip(bm,1,BMAX_PRIOR); return (1-1/b)/(1-1/BMAX_PRIOR)
def f_pinhole(lgM): return np.clip(0.22*(10**(lgM-8))**-0.307,0,1)   # Stone & Metzger 2016 eq. 29
def survive(lgM,prior):
    """Fraction of events surviving the Hills cut for a prior."""
    bm=bmax(lgM)
    if prior=="uniform": return np.clip((np.clip(bm,1,BMAX_PRIOR)-1)/(BMAX_PRIOR-1),0,1)
    if prior=="flc":     return tail_frac(bm)
    if prior=="sm16":    f=f_pinhole(lgM); return (1-f)+f*tail_frac(bm)  # delta at beta=1 + beta^-2 tail
NAMES={"uniform":"uniform β 1–30 (current)","flc":"β⁻² (full loss cone)","sm16":"Stone & Metzger 16: δ(β=1) + f_pinhole·β⁻²"}
COL={"uniform":C_UNI,"flc":C_FLC,"sm16":C_SM}
tag="use_sigma_neumayer_uniform_oleary21_tde_fallback"
det={}
for S in ["ZTF","LSST"]:
    det[S]=np.concatenate([h5py.File(f"{DATA}/TDE_SNRs_{S}_{tag}_{i}.h5")["lgMBH_mass"][:] for i in range(0,100,5)])
plt.rcParams.update({"font.size":11,"axes.facecolor":SURF,"figure.facecolor":SURF,"text.color":INK,"axes.labelcolor":INK2,
  "xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig,axs=plt.subplots(2,2,figsize=(15,9.5)); fig.subplots_adjust(hspace=.38,wspace=.22)
# (a) priors at fixed mass
ax=axs[0,0]; M0=6.5; b=np.linspace(1,30,600); bm=bmax(M0)
ax.plot(b,np.full_like(b,1/29),color=C_UNI,lw=2); ax.plot(b,1/b**2/(1-1/30),color=C_FLC,lw=2)
f=f_pinhole(M0); ax.plot(b,f/b**2/(1-1/30),color=C_SM,lw=2); ax.vlines(1,0,0.9,color=C_SM,lw=4); 
ax.text(1.4,0.88,f"{1-f:.0%} of events at β=1",color=INK,fontsize=9.5,va="top")
ax.axvline(bm,color=INK,lw=2); ax.axvspan(bm,30,color="#d8d7d2",alpha=.6,lw=0)
ax.text(bm+.4,0.55,f"Hills limit at lgMBH={M0}\nβ_max={bm:.1f}\n(star swallowed → no flare)",fontsize=10,va="top")
ax.set_xlim(0.8,30); ax.set_ylim(0,1.0); ax.set_yscale("symlog",linthresh=0.01); ax.set_ylim(0,1.0)
ax.set_xlabel("β"); ax.set_ylabel("prior density (shape)")
for k,y in (("uniform",0.0045),("flc",0.35),("sm16",0.13)): pass
ax.text(18,0.0045,"uniform",color=C_UNI,fontsize=10,va="bottom"); ax.text(9,0.014,"β⁻²",color=C_FLC,fontsize=10)
ax.text(4.2,0.15,"S&M16",color=C_SM,fontsize=10)
ax.set_title("A. The prior is the shape; Hills is the wall.\nImpact = prior mass that lands in the grey zone",loc="left",fontsize=12); ax.grid(color=GRID,lw=.6)
# (b) survival vs mass
ax=axs[0,1]; m=np.linspace(4,7,200)
for k in NAMES: ax.plot(m,survive(m,k),color=COL[k],lw=2.2,label=NAMES[k])
ax.set_ylim(0,1.05); ax.set_xlim(4,7); ax.set_xlabel("lgMBH"); ax.set_ylabel("fraction of events that survive Hills")
ax.set_title("B. Survival probability by mass, per β prior",loc="left",fontsize=12); ax.legend(frameon=False,fontsize=9.5,loc="lower left"); ax.grid(color=GRID,lw=.6)
# (c) detected mass histogram, LSST
ax=axs[1,0]; e=np.linspace(4,7,31); Md=det["LSST"]
ax.hist(Md,e,histtype="step",color=INK,lw=2,ls="--",label="no Hills cut (any prior: same)")
for k in NAMES: ax.hist(Md,e,weights=survive(Md,k),histtype="step",color=COL[k],lw=2.2,label="Hills cut + "+NAMES[k].split(" (")[0].split(":")[0])
ax.set_xlabel("lgMBH"); ax.set_ylabel("N detected (LSST)"); ax.legend(frameon=False,fontsize=9.5,loc="upper left"); ax.grid(color=GRID,lw=.6,axis="y")
ax.set_title("C. Detected mass distribution after the cut",loc="left",fontsize=12)
# (d) retained fraction of detected
ax=axs[1,1]; x=np.arange(2); w=.26; print("retained fraction of detected events:")
for j,k in enumerate(NAMES):
    vals=[survive(det[S],k).mean() for S in ["ZTF","LSST"]]; print(f"  {k:8s} ZTF {vals[0]:.3f} LSST {vals[1]:.3f}")
    bars=ax.bar(x+(j-1)*w,vals,w*.92,color=COL[k],label=NAMES[k].split(" (")[0].split(":")[0])
    for xi,v in zip(x+(j-1)*w,vals): ax.text(xi,v+.015,f"{v:.0%}",ha="center",fontsize=10)
ax.axhline(1,color=INK2,lw=1,ls="--"); ax.set_xticks(x); ax.set_xticklabels(["ZTF","LSST"]); ax.set_ylim(0,1.3)
ax.set_ylabel("fraction of detected TDEs kept"); ax.legend(frameon=False,fontsize=9.5,loc="upper center",ncol=3)
ax.set_title("D. How many detected TDEs the Hills cut removes",loc="left",fontsize=12)
fig.suptitle("tde_fallback population, use_sigma/neumayer/uniform (every 5th chunk). 1 M☉, 1 R☉ star, non-spinning BH. Valid because tde_fallback labels do not depend on β.",fontsize=10,color=INK2,x=.01,ha="left",y=.945)
plt.savefig(os.path.join(HERE,"hills_priors.png"),dpi=130,bbox_inches="tight")
