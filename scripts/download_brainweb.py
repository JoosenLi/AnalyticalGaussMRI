#!/usr/bin/env python
"""Download the paper's 0.5 mm BrainWeb subject through brainweb-dl."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, default=4)
    p.add_argument('--output', type=Path, default=Path('data/brainweb'))
    args = p.parse_args()
    from brainweb_dl import get_mri
    import nibabel as nib
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=True)
    data, affine = get_mri(args.subject, contrast='fuzzy', brainweb_dir=args.output,
                           with_affine=True)
    if data.shape[-1] != 12 or data.ndim != 4:
        raise ValueError(f'Expected twelve fuzzy channels; received {data.shape}')
    path = args.output / f'brainweb_s{args.subject:02d}_fuzzy.nii.gz'
    if not path.exists():
        nib.save(nib.Nifti1Image(np.asarray(data), affine), path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    record = {'subject': args.subject, 'shape': list(data.shape), 'affine': affine.tolist(),
              'sha256': h.hexdigest(), 'source': 'https://brainweb.bic.mni.mcgill.ca/',
              'downloader': 'brainweb-dl 0.4.6'}
    path.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(path)

if __name__ == '__main__':
    main()
