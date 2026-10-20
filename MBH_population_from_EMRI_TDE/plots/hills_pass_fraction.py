"""Pass fraction of the Hills cut as a function of MBH mass for the beta^-2 penetration prior."""
import os, h5py, numpy as np
os.chdir('./data')
Rsun=6.957e10; rS_sun=2.953e5   # cm; Schwarzschild radius of 1 Msun
def bmax(lgM, k=1.0):           # k=1: r_p>r_S (Stone&Metzger bmax=Rt/2Rg); k=2: r_p>2 r_S (capture orbit)
    """Maximum beta allowed by the Hills cut."""
    M=10**lgM; return Rsun*M**(1/3)/(k*rS_sun*M)
def frac_invb2(bm):             # fraction of beta^-2 on [1,30] with beta<=bm
    """Pass fraction for a beta^-2 prior."""
    b=np.clip(bm,1,30); return (1-1/b)/(1-1/30)
print("beta_max(lgM) k=1:",{m:round(float(bmax(m)),1) for m in [4,5,6,6.5,7]})
print("beta_max(lgM) k=2:",{m:round(float(bmax(m,2)),1) for m in [4,5,6,6.5,7]})
print("M where beta_max=1 (k=1): 10^%.2f ; k=2: 10^%.2f\n"%(np.log10((Rsun/rS_sun)**1.5),np.log10((Rsun/(2*rS_sun))**1.5)))
tr="astrophysical_data/../PRE_TRAIN_DATA_TDE/1E6_events"
print("=== TRAINING SETS (uniform beta 1-30): only cooling_envelope actually consumes beta ===")
for S in ["ZTF","LSST"]:
  for m in ["cooling_envelope","stream_stream_tde","tde_analytical","tde_fallback"]:
    p=h5py.File(f"{tr}/TDE_params_1E6_{S}_{m}.h5"); d=h5py.File(f"{tr}/TDE_detectability_1E6_{S}_{m}.h5")["detectability"][:]
    M=p["lgMBH"][:]; b=p["beta"][:]
    row=[]
    for k in (1,2):
      ok=b<=bmax(M,k)
      row.append((ok.mean(), ok[d==1].mean(), (d&ok).sum()/len(d), d.mean()))
    print(f"{S:4s} {m:18s} all-pass k1={row[0][0]:.3f} k2={row[1][0]:.3f} | detected-pass k1={row[0][1]:.3f} k2={row[1][1]:.3f} | det frac {row[0][3]:.3f}->k1 {row[0][2]:.3f} k2 {row[1][2]:.3f}")
print("\n=== TDE POPULATION (tde_fallback catalogues) ===")
import glob
base="astrophysical_data/100000"
combos=[(a,b,c) for a in ["use_sigma","use_lgMgal"] for b in ["neumayer","hannah"] for c in ["uniform","varying"]]
for S in ["ZTF","LSST"]:
  for (a,b_,c) in combos:
    tot=0;ok1=0;ok2=0;det=0;d1=0;d2=0;w=0
    for i in range(0,100,5):
      cat=h5py.File(f"{base}/all_TDE_params_data_{a}_{b_}_{c}_oleary21_tde_fallback_{i}.h5")
      s=h5py.File(f"{base}/TDE_SNRs_{S}_{a}_{b_}_{c}_oleary21_tde_fallback_{i}.h5")
      M=cat["lgMBH_mass"][:]; bt=cat["beta"][:]
      tot+=len(M); ok1+=(bt<=bmax(M,1)).sum(); ok2+=(bt<=bmax(M,2)).sum()
      Md=s["lgMBH_mass"][:]; bd=s["beta"][:]; det+=len(Md); d1+=(bd<=bmax(Md,1)).sum(); d2+=(bd<=bmax(Md,2)).sum()
    print(f"{S:4s} {a:10s} {b_:8s} {c:8s} all-pass k1={ok1/tot:.3f} k2={ok2/tot:.3f} | detected N={det} pass k1={d1/det:.3f} k2={d2/det:.3f}")
