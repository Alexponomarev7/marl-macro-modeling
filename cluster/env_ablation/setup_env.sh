#!/usr/bin/env bash
# One-time: a self-contained environment for the env ablation under ENV_PREFIX - Python 3.11 with the
# project's dependencies and CUDA torch, GNU Octave, and Dynare built from source for Octave. Needs
# internet, no root. Re-running skips the finished parts.
#
#   bash cluster/env_ablation/setup_env.sh                 # then: source $ENV_PREFIX/activate.sh
#
# ENV_PREFIX must be visible from the job pods (default: under /home/jovyan). About 8 GB.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-/home/jovyan/kovalenko/envs/marl-dynare}
DYNARE_VERSION=${DYNARE_VERSION:-7.1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
JOBS=${JOBS:-$(nproc)}

ENV=$ENV_PREFIX/env
MM=$ENV_PREFIX/bin/micromamba
export MAMBA_ROOT_PREFIX=$ENV_PREFIX/mamba
mkdir -p "$ENV_PREFIX/bin" "$ENV_PREFIX/build"

case $(uname -m) in
  x86_64) PLATFORM=linux-64 ;;
  aarch64) PLATFORM=linux-aarch64 ;;
  *) echo "unsupported machine $(uname -m)"; exit 1 ;;
esac

echo "== micromamba"
if [ ! -x "$MM" ]; then
  curl -fsSL -o "$MM" "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-$PLATFORM"
  chmod +x "$MM"
fi

echo "== conda env $ENV"
if [ ! -x "$ENV/bin/octave" ]; then
  "$MM" create -y -p "$ENV" -c conda-forge --override-channels \
    python=3.11 octave=10 \
    c-compiler cxx-compiler fortran-compiler make meson ninja pkg-config flex bison xz \
    libboost-devel gsl libmatio suitesparse "libblas=*=*openblas" liblapack \
    numpy pandas pyarrow scipy scikit-learn hydra-core omegaconf loguru tqdm python-dotenv pyyaml \
    gymnasium matplotlib pytest
fi
run() { "$MM" run -p "$ENV" "$@"; }

echo "== python packages"
if ! LD_LIBRARY_PATH=$ENV/lib run python -c "import torch, lightning, clearml, scipy.signal" 2>/dev/null; then
  run python -m pip install --no-cache-dir torch --index-url "$TORCH_INDEX"
  run python -m pip install --no-cache-dir "lightning>=2.6,<3" clearml
fi

echo "== Dynare $DYNARE_VERSION"
DYNARE_PREFIX=$ENV_PREFIX/dynare-$DYNARE_VERSION
if [ ! -f "$DYNARE_PREFIX/lib/dynare/matlab/dynare.m" ]; then
  cd "$ENV_PREFIX/build"
  [ -f "dynare-$DYNARE_VERSION.tar.xz" ] || curl -fsSLO "https://www.dynare.org/release/source/dynare-$DYNARE_VERSION.tar.xz"
  rm -rf "dynare-$DYNARE_VERSION"
  PATH=$ENV/bin:$PATH tar xJf "dynare-$DYNARE_VERSION.tar.xz"
  cd "dynare-$DYNARE_VERSION"
  # SLICOT is not on conda-forge; it only serves kalman_steady_state (estimation), which is then skipped
  sed -i "s/fortran_compiler.find_library('slicot_pic')/fortran_compiler.find_library('slicot_pic', required : false, disabler : true)/" meson.build
  # meson looks for Boost (header-only here) in system paths only
  run env BOOST_ROOT="$ENV" bash -c "meson setup build -Dbuild_for=octave --prefix='$DYNARE_PREFIX' --buildtype=release \
    && meson compile -C build -j $JOBS && meson install -C build"
fi

echo "== Octave Forge packages"
# Dynare needs datatypes and statistics; their latest releases need Octave >= 11 (conda-forge has 10), so
# these are the last ones for Octave 9-10. -global installs them into the env.
FORGE=(https://github.com/pr0m1th3as/datatypes/releases/download/release-1.2.0/datatypes-1.2.0.tar.gz
       https://github.com/gnu-octave/statistics/releases/download/release-1.8.2/statistics-1.8.2.tar.gz)
TRIPLE=$(basename "$(ls "$ENV"/bin/*-conda-linux-gnu-g++)" -g++)
octave_env() {
  # mkoctfile would otherwise call the compilers of conda-forge's build machine
  PATH=$ENV/bin:$PATH OCTAVE_HOME=$ENV CC=$ENV/bin/$TRIPLE-gcc CXX=$ENV/bin/$TRIPLE-g++ LD_CXX=$ENV/bin/$TRIPLE-g++ \
    F77=$ENV/bin/$TRIPLE-gfortran FC=$ENV/bin/$TRIPLE-gfortran octave-cli --eval "$1"
}
if ! octave_env "pkg load datatypes statistics" >/dev/null 2>&1; then
  cd "$ENV_PREFIX/build"
  for url in "${FORGE[@]}"; do
    [ -f "$(basename "$url")" ] || curl -fsSLO "$url"  # Octave's own downloader is less reliable
    octave_env "pkg install -global $(basename "$url")"
  done
fi
octave_env "pkg load datatypes statistics; disp('Forge packages OK')"

cat > "$ENV_PREFIX/activate.sh" <<EOF
# source this before running the ablation
export PATH=$ENV/bin:\$PATH
# the env's activation scripts: Octave needs OCTAVE_HOME (without it, it crashes or misses its own functions)
export CONDA_PREFIX=$ENV
_flags=\$-; set +eu  # the activation scripts read unset variables and may return non-zero
for f in $ENV/etc/conda/activate.d/*.sh; do . "\$f" || true; done
case \$_flags in *e*) set -e ;; esac; case \$_flags in *u*) set -u ;; esac
# pip's torch would otherwise load the system libstdc++, too old for the env's scipy
export LD_LIBRARY_PATH=$ENV/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export DYNARE_PATH=$DYNARE_PREFIX/lib/dynare/matlab
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONNOUSERSITE=1
EOF

echo "== smoke test"
# shellcheck disable=SC1091
source "$ENV_PREFIX/activate.sh"
SMOKE=$(mktemp -d)
cd "$REPO"
PYTHONPATH=$REPO python lib/dynare_traj2rl_transitions.py metadata.data_folder="$SMOKE" metadata.num_samples=2 \
  'metadata.only_models=[Hansen_1985,Gali_2010]' hydra.run.dir="$SMOKE/hydra" > "$SMOKE/log" 2>&1 \
  || { tail -50 "$SMOKE/log"; exit 1; }
N=$(find "$SMOKE/interim" -name "*.parquet" 2>/dev/null | wc -l)
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
if [ "$N" -lt 3 ]; then
  echo "SMOKE FAILED: $N/4 Dynare episodes; log: $SMOKE/log"; grep -m5 -iE "error|failed" "$SMOKE/log" || true; exit 1
fi
echo "OK: $N/4 Dynare episodes. Environment: source $ENV_PREFIX/activate.sh"
rm -rf "$SMOKE"
