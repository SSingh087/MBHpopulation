"""tde_fallback: how detection probability and detected numbers depend on bb (and the beta it maps to). Training set 1E6_events."""
import os; HERE = os.path.dirname(os.path.abspath(__file__))
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D='./data/PRE_TRAIN_DATA_TDE/1E6_events'
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e4e3df"; BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"
beta_of_bb=lambda bb: np.where(bb<1,0.6+1.25*bb,1.85+2.15*(bb-1))
data={}
for S in ["ZTF","LSST"]:
    p=h5py.File(f"{D}/TDE_params_1E6_{S}_tde_fallback.h5"); d=h5py.File(f"{D}/TDE_detectability_1E6_{S}_tde_fallback.h5")["detectability"][:]
    data[S]=(p["bb"][:],p["lgMBH"][:],d)
e=np.linspace(0,2,41); c=.5*(e[1:]+e[:-1])
def curve(bb,sel,d):
    n=np.histogram(bb[sel],e)[0]; k=np.histogram(bb[sel&(d==1)],e)[0]; pr=k/np.maximum(n,1); return pr,np.sqrt(pr*(1-pr)/np.maximum(n,1)),k
plt.rcParams.update({"font.size":11,"axes.facecolor":SURF,"figure.facecolor":SURF,"text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(18,5.4)); fig.subplots_adjust(wspace=.32,top=.78)
for S,col in (("ZTF",BLUE),("LSST",ORANGE)):
    bb,M,d=data[S]; pr,er,k=curve(bb,np.ones(len(bb),bool),d)
    ax[0].plot(c,pr,color=col,lw=2.2,label=S); ax[0].fill_between(c,pr-er,pr+er,color=col,alpha=.2,lw=0)
    ax[2].plot(c,k,color=col,lw=2.2,label=S)
for a in (ax[0],ax[2]):
    a.axvline(0.32,color=INK2,ls=":",lw=1.3); a.axvline(1,color=INK2,ls=":",lw=1.3)
ax[0].text(0.34,0.002,"β=1",fontsize=9.5,color=INK2); ax[0].text(1.02,0.002,"β=1.85",fontsize=9.5,color=INK2)
ax[0].set_xlabel("bb"); ax[0].set_ylabel("P(detected | bb)"); ax[0].legend(frameon=False); ax[0].grid(color=GRID,lw=.6)
ax[0].set_title("A. Detection probability vs bb\n(shaded: binomial 1σ)",loc="left",fontsize=12)
bb,M,d=data["LSST"]; edges=[(4,5),(5,5.7),(5.7,6.3),(6.3,7.01)]; cols=[BLUE,ORANGE,AQUA,"#6250d6"]
for (a,b2),col in zip(edges,cols):
    sel=(M>=a)&(M<b2); pr,er,k=curve(bb,sel,d); ax[1].plot(c,pr,color=col,lw=2.2,label=f"lgMBH {a}–{min(b2,7):g}")
ax[1].set_xlabel("bb"); ax[1].set_ylabel("P(detected | bb, mass)   LSST"); ax[1].legend(frameon=False,fontsize=10); ax[1].grid(color=GRID,lw=.6)
ax[1].set_title("B. bb dependence is strongest at high mass",loc="left",fontsize=12)
ax[2].set_xlabel("bb"); ax[2].set_ylabel("N detected per bb bin (0.05)"); ax[2].legend(frameon=False); ax[2].grid(color=GRID,lw=.6)
ax[2].set_title("C. Detected numbers in the training set\n(250k events, flat in bb)",loc="left",fontsize=12)
# secondary axis on A with beta
sec=ax[0].secondary_xaxis("top",functions=(lambda x: beta_of_bb(np.asarray(x,float)),lambda y: np.where(y<1.85,(np.asarray(y,float)-0.6)/1.25,1+(np.asarray(y,float)-1.85)/2.15)))
sec.set_xlabel("β",color=INK2)
tot={S:(data[S][2].sum()) for S in data}
fig.suptitle("tde_fallback, 1 M☉ star.  Detected totals: ZTF %d, LSST %d.   Detection rises steeply for bb<0.3 (β<1), more slowly to bb≈0.9 (β≈1.7), then is flat."%(tot["ZTF"],tot["LSST"]),fontsize=10.5,color=INK2,x=.01,ha="left",y=.96)
plt.savefig(os.path.join(HERE,"bb_detection.png"),dpi=130,bbox_inches="tight")
for S in data:
    bb,M,d=data[S]; lo=bb<0.32; print(S,"P(det): bb<0.32 %.3f | 0.32-0.6 %.3f | >0.6 %.3f ; share of detections from bb<0.32: %.1f%%"%(d[lo].mean(),d[(bb>=.32)&(bb<.6)].mean(),d[bb>=.6].mean(),100*d[lo].sum()/d.sum()))
