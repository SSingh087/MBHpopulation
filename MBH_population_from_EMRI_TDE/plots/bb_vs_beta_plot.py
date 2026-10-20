"""tde_fallback: how bb maps to beta (redback _tde_mosfit_engine, gamma=4/3 for a 1 Msun star) and what uniform bb implies for beta."""
import os; HERE = os.path.dirname(os.path.abspath(__file__))
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D='./data/PRE_TRAIN_DATA_TDE/1E6_events'
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e4e3df"; BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"
beta_of_bb=lambda bb: np.where(bb<1,0.6+1.25*bb,1.85+2.15*(bb-1))
bb=h5py.File(f"{D}/TDE_params_1E6_LSST_tde_fallback.h5")["bb"][:]; b=beta_of_bb(bb)
plt.rcParams.update({"font.size":11,"axes.facecolor":SURF,"figure.facecolor":SURF,"text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(17,5.2)); fig.subplots_adjust(wspace=.3,top=.78)
x=np.linspace(0,2,400); ax[0].plot(x,beta_of_bb(x),color=BLUE,lw=2.5)
ax[0].axhspan(0.6,1.0,color="#d8d7d2",alpha=.5,lw=0); ax[0].axhline(1,color=INK2,lw=1,ls=":")
ax[0].plot([0.32],[1.0],"o",color=INK); ax[0].annotate("β = 1 at bb = 0.32",(0.32,1.0),(0.55,1.45),fontsize=10,arrowprops=dict(arrowstyle="-",color=INK2))
ax[0].plot([1],[1.85],"o",color=INK); ax[0].annotate("kink at bb = 1\nslope 1.25 → 2.15",(1,1.85),(1.15,1.2),fontsize=10,arrowprops=dict(arrowstyle="-",color=INK2))
ax[0].text(1.05,0.62,"β<1: partial disruption zone",fontsize=9.5,color=INK2); ax[0].text(0.02,0.62,"",fontsize=9)
ax[0].set_xlabel("bb  (what the pipeline samples, uniform 0–2)"); ax[0].set_ylabel("β  (what the model uses)"); ax[0].set_ylim(0.5,4.2); ax[0].grid(color=GRID,lw=.6)
ax[0].set_title("A. bb → β (piecewise linear)",loc="left",fontsize=12)
e=np.linspace(0.6,4.0,35); ax[1].hist(b,e,density=True,color=BLUE,alpha=.9,label="β from uniform bb (current)")
bg=np.linspace(1,4,200); ax[1].plot(bg,bg**-2/0.75,color=ORANGE,lw=2.4,label="β⁻² on [1,4] (full loss cone)")
f=np.clip(0.22*(10**(6.5-8))**-0.307,0,1); ax[1].plot(bg,f*bg**-2/0.75,color=AQUA,lw=2.4,label=f"S&M16 tail, f_pinhole={f:.2f} at lgMBH 6.5\n(+ {1-f:.0%} of events at β = 1)")
ax[1].axvspan(0.6,1.0,color="#d8d7d2",alpha=.5,lw=0); ax[1].set_xlabel("β"); ax[1].set_ylabel("probability density"); ax[1].legend(frameon=False,fontsize=9.5); ax[1].grid(color=GRID,lw=.6)
ax[1].set_title("B. Uniform bb gives a nearly flat β prior,\nnot Luca's peak at β = 1",loc="left",fontsize=12)
ax[2].hist(bb,np.linspace(0,2,41),color=BLUE,alpha=.9); ax[2].axvline(0.32,color=INK2,ls=":",lw=1.5); ax[2].text(0.34,ax[2].get_ylim()[1]*0.5,"β = 1",fontsize=10,color=INK2)
ax[2].set_xlabel("bb"); ax[2].set_ylabel("N events (training set, LSST)"); ax[2].grid(color=GRID,lw=.6,axis="y")
ax[2].set_title("C. What the pipeline actually samples:\nflat in bb",loc="left",fontsize=12)
fig.suptitle("tde_fallback, 1 M☉ star (γ = 4/3).  Fraction of events with β < 1: %.0f%%, β > 1.85: %.0f%%, β > 3: %.0f%%"%(100*(b<1).mean(),100*(b>1.85).mean(),100*(b>3).mean()),fontsize=10.5,color=INK2,x=.01,ha="left",y=.95)
plt.savefig(os.path.join(HERE,"bb_vs_beta.png"),dpi=130,bbox_inches="tight"); print((b<1).mean(),(b>1.85).mean(),(b>3).mean())
