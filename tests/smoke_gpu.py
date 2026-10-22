"""Small GPU integration test, runnable against an installed wheel from any directory."""
import json
import sys
import torch
from gaussmrisim.conventions import FOV
from gaussmrisim.datasets.brainweb import VolumeTarget
from gaussmrisim.fitting.fitter import FitConfig, fit, render_full
from gaussmrisim.fourier.cuda_ft import kspace_cuda
from gaussmrisim.fourier.gaussian_ft import kspace
from gaussmrisim.coils.fourier_csm import FourierCSM, cartesian_mode_grid, coil_kspace
from gaussmrisim.motion.rigid import RigidPoses, moved_coil_kspace, rotation_from_euler
from gaussmrisim.representation.gaussian3d import random_mixture


def main():
    assert torch.cuda.is_available(), 'CUDA is required; no CPU fallback.'
    dev = 'cuda:0'
    torch.cuda.set_device(dev)
    torch.manual_seed(0)
    fov = FOV((16.,)*3,(16,)*3)
    pts = fov.grid_coords(device=dev)
    # Independent, non-Gaussian fitting target.
    vol = ((pts[:,0]/5)**2 + (pts[:,1]/4)**2 + (pts[:,2]/3)**2 < 1).float().reshape(16,16,16)
    tgt = VolumeTarget(vol, fov, 'ellipsoid_smoke')
    cfg = FitConfig(n_primitives=64,steps=12,patch=16,densify_every=100,
                    tv_weight=1e-4,tv_samples=64,aniso_init=0.)
    mix, trace = fit(tgt,cfg,verbose=False)
    img = render_full(mix,tgt)
    assert torch.isfinite(img).all() and img.max()>0
    assert mix.mu.is_cuda and mix.n>0
    k = torch.randn(31,3,device=dev)*0.08
    g = random_mixture(48,fov_mm=(12.,)*3,sigma_range=(1.,2.),seed=7,device=dev)
    ref = kspace(g,k); got = kspace_cuda(g,k)
    err = float((got-ref).norm()/ref.norm()); assert err<1e-5,err
    freqs = cartesian_mode_grid(3,fov.fov_mm,device=dev)
    beta = torch.randn(2,len(freqs),device=dev,dtype=torch.complex64)/len(freqs)
    csm = FourierCSM(freqs,beta)
    ref = coil_kspace(g,k,csm); got = kspace_cuda(g,k,csm)
    coilerr = float((got-ref).norm()/ref.norm()); assert coilerr<2e-5,coilerr
    R=rotation_from_euler(torch.tensor([[3.,-2.,1.]],device=dev),device=dev)
    b=torch.tensor([[0.4,-0.2,0.3]],device=dev)
    poses=RigidPoses(R=R.expand(len(k),-1,-1),t=b.expand(len(k),-1))
    moving=moved_coil_kspace(g,k,csm,poses)
    ref = kspace_cuda(g.rigid_transform(R[0], b[0]), k, csm)
    motionerr = float((moving-ref).norm()/ref.norm())
    assert motionerr < 2e-5, motionerr
    assert torch.isfinite(moving).all()
    print(json.dumps({'gpu':torch.cuda.get_device_name(0),'fit_primitives':mix.n,
                      'fourier_relative_error':err,'coil_relative_error':coilerr,'motion_relative_error':motionerr,
                      'status':'PASS'},indent=2))

if __name__=='__main__': main()
