#!/usr/bin/env python
"""Run a paper experiment; use --dry-run to inspect the exact commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FIT = 'outputs/fits/brainweb_s04_fuzzy_0.50mm_measured_area'


def commands(stage, device):
    def cmd(script, *args):
        folder = 'scripts' if script == 'download_brainweb' else 'experiments'
        return [sys.executable, str(ROOT/folder/f'{script}.py'), *map(str, args)]
    if stage == 'download':
        return [cmd('download_brainweb')]
    if stage == 'fit':
        return [cmd('fit_tissues_brainweb', '--device', device, '--subject', 4,
                    '--budget-file', 'configs/brainweb_budget.json', '--steps', 1500,
                    '--tag', 'measured', '--rule', 'area')]
    if stage == 'shepplogan':
        out = []
        for subject in ('const', 'natural'):
            common = ['--subject', subject, '--device', device]
            out += [cmd('shepplogan_benchmark', 'truth', *common),
                    cmd('shepplogan_benchmark', 'fit', *common, '--total', 300000,
                        '--floor', 200, '--steps', 1500, '--tv-weight', 0.0001, '--tv-samples', 8192),
                    cmd('shepplogan_benchmark', 'cells', *common, '--h', 0),
                    cmd('shepplogan_benchmark', 'eval', *common),
                    cmd('shepplogan_tets', *common, '--vol', 5, '--smooth', 0, '--step', 1),
                    cmd('shepplogan_tets', *common, '--vol', 8, '--smooth', 0, '--step', 2)]
        return out
    if stage == 'discretization':
        return [cmd('aliasing_box_baseline', '--device', device, '--traj', 'cartesian', '--h', h, '--ss', ss)
                for h, ss in [(1,4),(0.5,2)]] + [
                cmd('aliasing_cg_radial', '--device', device, '--h', h, '--ss', ss,
                    '--fixed', 30, '--primary', 30, '--dc-init') for h, ss in [(1,4),(0.5,2)]]
    if stage == 'gibbs': return [cmd('gibbs_demo', '--device', device)]
    if stage == 'coils': return [cmd('bench_parallel_radial', '--device', device)]
    if stage == 'motion':
        return [cmd('motion_aphabamas', '--device', device, '--traj', *range(8), '--no-nufft')]
    if stage == 'figures': return [cmd('figures', 'shepplogan', 'gibbs', 'noisefloor', 'motion')]
    raise ValueError(stage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['download','fit','shepplogan','discretization','gibbs','coils','motion','figures','all'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not args.device.startswith('cuda:'): p.error('Experiments require a CUDA device.')
    stages = ['download','fit','shepplogan','discretization','gibbs','coils','motion','figures'] if args.stage == 'all' else [args.stage]
    cmds = [c for s in stages for c in commands(s, args.device)]
    for c in cmds: print(shlex.join(c), flush=True)
    if args.dry_run: return
    os.chdir(ROOT)
    os.environ["GMRISIM_DEVICE"] = args.device
    # Set device for plotting helpers as well as explicit --device drivers.
    if args.stage != 'download':
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; CPU fallback is disabled.')
        torch.cuda.set_device(args.device)
    logs = ROOT/'outputs/logs'; logs.mkdir(parents=True, exist_ok=True)
    record = {'stage': args.stage, 'commands': cmds, 'started_unix': time.time(), 'steps': [],
              'config_sha256': hashlib.sha256((ROOT/'configs/brainweb_budget.json').read_bytes()).hexdigest()}
    manifest = logs/f'{args.stage}-{time.time_ns()}.json'
    try:
        for i,c in enumerate(cmds):
            log = logs/f'{manifest.stem}-{i:02d}.log'
            with log.open('w') as f:
                r = subprocess.run(c, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            record['steps'].append({'command':c,'returncode':r.returncode,'log':str(log.relative_to(ROOT))})
            if r.returncode: raise RuntimeError(f'Command failed; inspect {log}')
    finally:
        record['finished_unix'] = time.time()
        manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(f'Completed. Provenance: {manifest}')

if __name__ == '__main__': main()
