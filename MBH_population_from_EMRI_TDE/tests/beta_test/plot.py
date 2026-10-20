"""Plot the beta results of beta_test.py."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=np.load('results.npz')
S=['stone_metzger','full_loss_cone','empty_loss_cone']
fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
for a,sv in zip(ax,['ZTF','LSST']):
    ms=['cooling_envelope','stream_stream_tde','tde_analytical','tde_fallback']
    for j,s in enumerate(S):
        r=[]; e=[]
        for m in ms:
            u=d[f'{sv}|{m}|uniform']; x=d[f'{sv}|{m}|{s}']/u
            r.append(x.mean()); e.append(x.std())
        a.errorbar(np.arange(4)+0.2*(j-1),r,e,fmt='o',capsize=3,label=s)
    a.axhline(1,c='k',ls='--'); a.set_xticks(range(4)); a.set_xticklabels(ms,rotation=20)
    a.set_title(sv)
ax[0].set_ylabel('detections / uniform-β detections'); ax[0].legend()
plt.tight_layout(); plt.savefig('beta_test.png',dpi=150)
