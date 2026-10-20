"""Hills-cut diagnostic: beta versus MBH mass, detected-mass histograms and pass fraction versus mass."""
import os; HERE = os.path.dirname(os.path.abspath(__file__))
import os, h5py, numpy as np
DATA='./data'
os.chdir(f'{DATA}/astrophysical_data/100000')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e4e3df"
BLUE,ORANGE="#2a78d6","#eb6834"
Rsun=6.957e10; rS=2.953e5
bmax=lambda lgM,k=1: Rsun*(10**lgM)**(1/3)/(k*rS*10**lgM)
base="."; tag="use_sigma_neumayer_uniform_oleary21_tde_fallback"
Ma=[];Ba=[];Md=[];Bd=[]
for i in range(0,100,5):
    c=h5py.File(f"all_TDE_params_data_{tag}_{i}.h5"); Ma.append(c["lgMBH_mass"][:]); Ba.append(c["beta"][:])
    s=h5py.File(f"TDE_SNRs_LSST_{tag}_{i}.h5"); Md.append(s["lgMBH_mass"][:]); Bd.append(s["beta"][:])
Ma,Ba,Md,Bd=map(np.concatenate,(Ma,Ba,Md,Bd))
ok_a=Ba<=bmax(Ma); ok_d=Bd<=bmax(Md)
plt.rcParams.update({"font.size":11,"axes.facecolor":SURF,"figure.facecolor":SURF,"text.color":INK,"axes.labelcolor":INK2,
  "xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,7.5)); gs=GridSpec(2,2,width_ratios=[1.25,1],hspace=0.38,wspace=0.22)
# A: beta vs mass
ax=fig.add_subplot(gs[:,0]); m=np.linspace(4,7,200)
ax.set_yscale("log"); ax.set_ylim(1,30); ax.set_xlim(4,7)
ax.fill_between(m,bmax(m),30,color="#d8d7d2",alpha=.55,lw=0,zorder=0)
rng=np.random.default_rng(0); idx=rng.choice(len(Md),min(12000,len(Md)),replace=False)
good=ok_d[idx]
ax.scatter(Md[idx][good],Bd[idx][good],s=5,c=ORANGE,alpha=.30,lw=0,label="detected, survives Hills cut")
ax.scatter(Md[idx][~good],Bd[idx][~good],s=5,c=INK2,alpha=.35,lw=0,label="detected, but star would be swallowed")
ax.plot(m,bmax(m),color=INK,lw=2); ax.plot(m,bmax(m,2),color=INK,lw=1.5,ls="--")
ax.text(4.08,2.2,"Hills limit  β_max = r_t / r_S\n(star swallowed whole above this line)",color=INK,fontsize=10,va="bottom")
ax.text(5.55,3.0,"stricter: r_p > 2 r_S",color=INK2,fontsize=9,rotation=-33)
ax.set_xlabel("lgMBH"); ax.set_ylabel("β  (log scale)")
ax.set_title(f"A. The catalogue puts detected TDEs where no flare can exist\n{(~ok_d).mean():.0%} of LSST-detected events lie above the Hills line",loc="left",fontsize=12)
ax.legend(loc="lower right",frameon=False,markerscale=3,fontsize=10)
ax.grid(color=GRID,lw=.6)
# B: histograms of detected mass
ax=fig.add_subplot(gs[0,1]); e=np.linspace(4,7,31)
ax.hist(Md,e,color=BLUE,alpha=.9,label="detected (current)")
ax.hist(Md[ok_d],e,color=ORANGE,alpha=.95,label="detected AND passes Hills")
ax.set_xlabel("lgMBH"); ax.set_ylabel("N detected (LSST)")
ax.set_title("B. Hills cut removes the high-mass pile-up",loc="left",fontsize=12); ax.legend(frameon=False,fontsize=10,loc="upper left"); ax.grid(color=GRID,lw=.6,axis="y")
# C: pass fraction vs mass; current prior vs beta^-2
ax=fig.add_subplot(gs[1,1]); e=np.linspace(4,7,16); c=.5*(e[1:]+e[:-1])
fr=[ok_a[(Ma>=a)&(Ma<b)].mean() for a,b in zip(e[:-1],e[1:])]
ax.plot(c,fr,"o-",color=BLUE,lw=2,ms=5)
mm=np.linspace(4,7,100); bm=np.clip(bmax(mm),1,30)
ax.plot(mm,(1-1/bm)/(1-1/30),color=ORANGE,lw=2)
ax.text(4.1,0.80,"β ∝ β⁻²  (full loss cone)",color=INK,fontsize=10,va="top")
ax.text(5.65,0.62,"current: β uniform 1–30",color=INK,fontsize=10)
ax.set_ylim(0,1.05); ax.set_xlim(4,7); ax.set_xlabel("lgMBH"); ax.set_ylabel("fraction passing Hills")
ax.set_title("C. Survival probability vs mass depends on the β prior",loc="left",fontsize=12); ax.grid(color=GRID,lw=.6)
fig.suptitle("LSST population, tde_fallback, use_sigma / neumayer / uniform  (every 5th chunk). Star fixed at 1 M☉, 1 R☉, non-spinning BH.",fontsize=10,color=INK2,x=0.01,ha="left",y=0.995)
out=os.path.join(HERE,'hills_diagnostic.png'); plt.savefig(out,dpi=130,bbox_inches="tight"); print(out,(~ok_d).mean(),(~ok_a).mean())
