import sys; sys.path.insert(0,'.')
from egsf.data.jdb_r import generate_jdbr
import numpy as np, torch
from egsf.models.bf import train_bf_mod
from egsf.reliance.d0_reliance import compute_d0_reliance

ds = generate_jdbr(seed=0)

bf, _ = train_bf_mod(
    [ds['train']['X1'],ds['train']['X2']], ds['train']['y'],
    [ds['val_id']['X1'],ds['val_id']['X2']], ds['val_id']['y'],
    in_dims=[8,8], num_classes=4, seed=0, verbose=False
)
bf.eval()

def acc(X1,X2,y):
    with torch.no_grad():
        out = bf([torch.tensor(X1,dtype=torch.float32),torch.tensor(X2,dtype=torch.float32)])
        logits = out[0] if isinstance(out,tuple) else out
        return np.mean(logits.argmax(-1).numpy()==y)

print("BF test_id:      ", round(acc(ds['test_id']['X1'],ds['test_id']['X2'],ds['test_id']['y']),4))
print("BF test_conflict:", round(acc(ds['test_conflict']['X1'],ds['test_conflict']['X2'],ds['test_conflict']['y']),4), " (should collapse ~0.25)")

rel_id = compute_d0_reliance(bf, [ds['test_id']['X1'],ds['test_id']['X2']], seed=0)
rel_conf = compute_d0_reliance(bf, [ds['test_conflict']['X1'],ds['test_conflict']['X2']], seed=0)
print(f"D0 ID rel:   X1={rel_id[:,0].mean():.4f} X2={rel_id[:,1].mean():.4f}  (X2 should be dominant)")
print(f"D0 conf rel: X1={rel_conf[:,0].mean():.4f} X2={rel_conf[:,1].mean():.4f}")
