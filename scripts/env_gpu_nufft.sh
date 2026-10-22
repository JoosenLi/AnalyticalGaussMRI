# Source AFTER activating your Python environment. Set CUDA_HOME to your toolkit.
# Example: export CUDA_HOME=/usr/local/cuda
if [ -n "${CUDA_HOME:-}" ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
_gmri_libs=$(python - <<'PY'
import site
from pathlib import Path
roots = site.getsitepackages()
names = ('cublas', 'cusolver', 'cusparse', 'cufft', 'curand', 'nvjitlink', 'cuda_runtime', 'cuda_nvrtc', 'cuda_cupti')
print(':'.join(str(Path(root)/'nvidia'/name/'lib') for root in roots for name in names if (Path(root)/'nvidia'/name/'lib').is_dir()))
PY
)
export LD_LIBRARY_PATH="${_gmri_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _gmri_libs
