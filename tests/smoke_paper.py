"""Tiny paper-dependency checks; no anatomy download or full experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
import torch
from aliasing_recon_noise import GpuNufft
from aliasing_cg_radial import lsqr_iterates
from verify_polyhedral import box
from gaussmrisim.fourier import polyhedral_ft as P
from gaussmrisim.fourier.polyhedral_tets import tetrahedralize, tet_volumes

assert torch.cuda.is_available(), 'CUDA required'
torch.cuda.set_device(0)
import cupy as cp
cp.cuda.Device(0).use()
rng=np.random.default_rng(8)
k=rng.uniform(-.4,.4,(64,3))
x=rng.normal(size=(8,8,8)).astype(np.complex64)
nu=GpuNufft(k,1.,x.shape)
z,y,xx=np.meshgrid(np.arange(8)-4,np.arange(8)-4,np.arange(8)-4,indexing='ij')
pts=np.stack([xx,y,z],axis=-1).reshape(-1,3)
ref=np.exp(-2j*np.pi*(k@pts.T))@x.ravel()
got=nu.forward(x)
err=np.linalg.norm(got-ref)/np.linalg.norm(ref)
assert err<2e-5,err
b=cp.asarray(got/nu.norm,dtype=cp.complex64)
it=lsqr_iterates(nu.op,b,[2],x.shape,x0=None)
assert np.isfinite(it[2]).all()
shift=torch.tensor([0.,0.,0.],dtype=torch.float64)
mesh=box(1.,1.,1.,shift)
rec=P.pack([mesh],[1.],device='cuda:0')
kt=torch.tensor(k,device='cuda:0',dtype=torch.float64)
out=P.polyhedral_ft(rec,kt,'double')
truth=8*torch.sinc(2*kt).prod(-1)
mesherr=float((out-truth).abs().max()/8)
assert mesherr<1e-10,mesherr
V,T=tetrahedralize(mesh,max_volume=1.)
assert abs(float(tet_volumes(V,T).sum())-8)<1e-8
print({'status':'PASS','nufft_direct_dft_relative_error':float(err),
       'polyhedral_box_error':mesherr,'lsqr_iterations':2,'tetrahedra':len(T)})
