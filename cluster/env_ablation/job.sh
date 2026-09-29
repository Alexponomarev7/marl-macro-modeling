#!/usr/bin/env bash
# Job entrypoint (submit.py): the ablation, or with SETUP=1 the one-time environment build.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
if [ "${SETUP:-0}" = 1 ]; then
  exec bash cluster/env_ablation/setup_env.sh
fi
# shellcheck disable=SC1091
source "$ENV_PREFIX/activate.sh"
nvidia-smi || true
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print(torch.cuda.get_device_name())"
exec python cluster/env_ablation/run.py
