"""tde_fallback only: what matters, the penetration prior or the Hills cut?
tde_fallback's penetration factor comes from `bb` (redback _tde_mosfit_engine, gamma=4/3 since mstar=1):
   bb in [0,1): beta = 0.6 + 1.25 bb ;  bb in [1,2]: beta = 1.85 + 2.15 (bb-1)   -> beta in [0.6, 4.0]
Detectability labels DO depend on bb, so changing the beta prior changes the detection rate (unlike the unused catalogue `beta`)."""
import os; HERE = os.path.dirname(os.path.abspath(__file__))
import h5py, numpy as np
D='./data/PRE_TRAIN_DATA_TDE/1E6_events'
Rsun=6.957e10; rS=2.953e5
bmax=lambda l: Rsun*(10**l)**(1/3)/(rS*10**l)
def beta_of_bb(bb): return np.where(bb<1,0.6+1.25*bb,1.85+2.15*(bb-1))
def f_pin(l): return np.clip(0.22*(10**(l-8))**-0.307,0,1)
BE=np.concatenate([[0.6,0.8,0.9,1.1],np.arange(1.3,3.95,0.2),[4.0]])   # beta bins; one bin is exactly [0.9,1.1] around beta=1
ME=np.linspace(4,7,13)          # mass bins
for S in ["ZTF","LSST"]:
    p=h5py.File(f"{D}/TDE_params_1E6_{S}_tde_fallback.h5"); d=h5py.File(f"{D}/TDE_detectability_1E6_{S}_tde_fallback.h5")["detectability"][:]
    M=p["lgMBH"][:]; b=beta_of_bb(p["bb"][:])
    print(f"\n===== {S} tde_fallback  (N={len(M)}, detected={d.sum()}) =====")
    print("beta range",b.min().round(2),b.max().round(2),"| Hills pass: all",(b<=bmax(M)).mean(),"detected",(b[d==1]<=bmax(M[d==1])).mean(),
          "| lgM where beta_max=4.0:",round(7+1.5*np.log10(bmax(7)/4.0),2))
    # detection fraction in (mass, beta) bins
    H=np.histogram2d(M,b,[ME,BE])[0]; Hd=np.histogram2d(M[d==1],b[d==1],[ME,BE])[0]
    det=np.divide(Hd,H,out=np.zeros_like(Hd),where=H>0)
    bc=.5*(BE[1:]+BE[:-1]); mc=.5*(ME[1:]+ME[:-1])
    print("beta bin centres:",np.round(bc,2)); print("detection fraction vs beta (all masses):",np.round([Hd[:,j].sum()/H[:,j].sum() for j in range(len(bc))],3))
    # priors in beta-bin space (probability per bin), cut applied via beta<=beta_max(M)
    BW=np.diff(BE)
    def prior(name,M0):
        w=np.zeros(len(bc))
        hi=4.0
        if name=="current":   # bb uniform -> density in beta
            w=np.array([ (np.clip(BE[j+1],0.6,1.85)-np.clip(BE[j],0.6,1.85))/1.25*0.5 + (np.clip(BE[j+1],1.85,4.0)-np.clip(BE[j],1.85,4.0))/2.15*0.5 for j in range(len(bc))])
        elif name=="beta=1 (empty loss cone)": w[np.argmin(abs(bc-1.0))]=1
        elif name=="beta^-2 on [1,4]":
            lo=np.clip(BE[:-1],1,hi);up=np.clip(BE[1:],1,hi); w=(1/lo-1/up)
        elif name=="S&M16: delta(1)+f*beta^-2":
            lo=np.clip(BE[:-1],1,hi);up=np.clip(BE[1:],1,hi); t=(1/lo-1/up); t=t/(1-1/hi)
            w=f_pin(M0)*t; w[np.argmin(abs(bc-1.0))]+= (1-f_pin(M0))   # point mass placed in the bin containing beta=1
        return w/w.sum()
    names=["current","beta=1 (empty loss cone)","beta^-2 on [1,4]","S&M16: delta(1)+f*beta^-2"]
    res={}
    for nm in names:
        for hills in (False,True):
            dm=np.zeros(len(mc))
            for i,m0 in enumerate(mc):
                w=prior(nm,m0)
                if hills: w=w*(bc<=bmax(m0)); 
                dm[i]=(w*det[i]).sum()            # P(detect | M) under this prior (+cut = event lost)
            res[(nm,hills)]=dm
    base=res[("current",False)]
    pop=np.histogram(M,ME)[0]
    print(f"{'prior':32s} {'overall det frac':>17s} {'rel. to current':>16s} | det frac at lgM 4.25 / 5.5 / 6.75   (Hills cut changes?)")
    for nm in names:
        dm=res[(nm,False)]; dmh=res[(nm,True)]
        ov=(pop*dm).sum()/pop.sum(); ovh=(pop*dmh).sum()/pop.sum(); ob=(pop*base).sum()/pop.sum()
        print(f"{nm:32s} {ov:17.4f} {ov/ob:16.3f} | {dm[0]:.3f} {dm[5]:.3f} {dm[11]:.3f}   Hills: {ovh:.4f}")

    out_all=globals().setdefault("OUT",{}); out_all[S]=dict(bc=bc,mc=mc,dfb=[Hd[:,j].sum()/H[:,j].sum() for j in range(len(bc))],res=res,pop=pop,names=names,base=base,M=M,d=d)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e4e3df"
COL={"current":"#2a78d6","beta^-2 on [1,4]":"#eb6834","S&M16: delta(1)+f*beta^-2":"#1baf7a","beta=1 (empty loss cone)":INK2}
plt.rcParams.update({"font.size":11,"axes.facecolor":SURF,"figure.facecolor":SURF,"text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(18,5.6)); fig.subplots_adjust(wspace=.32,top=.80)
for S,ls in (("ZTF","-"),("LSST","--")):
    o=OUT[S]; ax[0].plot(o["bc"],o["dfb"],ls,color=INK,lw=2,marker="o",ms=3,label=S)
ax[0].axvspan(0.6,1.0,color="#d8d7d2",alpha=.5,lw=0); ax[0].text(0.62,0.003,"β<1: partial\ndisruption",fontsize=9)
ax[0].set_xlabel("β  (from bb)"); ax[0].set_ylabel("detection fraction"); ax[0].legend(frameon=False)
ax[0].set_title("A. Detection depends strongly on β below ~1.4,\nthen flat: so the prior matters through the labels",loc="left",fontsize=11.5); ax[0].grid(color=GRID,lw=.6)
for k,(nm) in enumerate(["current","beta=1 (empty loss cone)","beta^-2 on [1,4]","S&M16: delta(1)+f*beta^-2"]):
    o=OUT["LSST"]; ax[1].plot(o["mc"],o["res"][(nm,False)],color=COL[nm],lw=2.2,ls="--" if "empty" in nm else "-",label=nm)
m=np.linspace(4,7,50)
ax[1].set_xlabel("lgMBH"); ax[1].set_ylabel("P(detect | mass)   LSST"); ax[1].legend(frameon=False,fontsize=9,loc="upper left"); ax[1].grid(color=GRID,lw=.6)
ax[1].set_title("B. β prior changes P(detect), mostly at high mass.\nHills cut: no change (curves identical)",loc="left",fontsize=11.5)
names=["current","beta=1 (empty loss cone)","beta^-2 on [1,4]","S&M16: delta(1)+f*beta^-2"]; x=np.arange(2); w=.2
for j,nm in enumerate(names):
    vals=[]
    for S in ["ZTF","LSST"]:
        o=OUT[S]; vals.append((o["pop"]*o["res"][(nm,False)]).sum()/(o["pop"]*o["base"]).sum())
    ax[2].bar(x+(j-1.5)*w,vals,w*.92,color=COL[nm],label=nm)
    for xi,v in zip(x+(j-1.5)*w,vals): ax[2].text(xi,v+.015,f"{v:.2f}",ha="center",fontsize=9.5)
hv=[ (OUT[S]["pop"]*OUT[S]["res"][("current",True)]).sum()/(OUT[S]["pop"]*OUT[S]["res"][("current",False)]).sum() for S in ["ZTF","LSST"]]
ax[2].axhline(1,color=INK2,lw=1,ls=":"); ax[2].set_xticks(x); ax[2].set_xticklabels(["ZTF","LSST"]); ax[2].set_ylim(0,1.35)
ax[2].set_ylabel("detected TDEs relative to current prior"); ax[2].legend(frameon=False,fontsize=8.5,loc="upper center",ncol=2)
ax[2].set_title(f"C. Prior effect: 0.58–1.04x of current.\nHills effect: {hv[0]:.2f}x (ZTF), {hv[1]:.2f}x (LSST)",loc="left",fontsize=11.5)
fig.suptitle("tde_fallback only (1 M☉ star, β from bb ∈ [0,2] → β ∈ [0.6, 4.0]); training set 1E6 events; Hills: r_p > r_S",fontsize=10,color=INK2,x=.01,ha="left",y=.96)
plt.savefig(os.path.join(HERE,"hills_prior_tde_fallback.png"),dpi=130,bbox_inches="tight")
