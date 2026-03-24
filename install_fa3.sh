# 0) Sanity checks
nvidia-smi
nvcc --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# 1) Build deps
python -m pip install -U pip setuptools wheel
python -m pip install packaging psutil ninja

# 2) Clone the repo
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper

# 3) Build FA3
# Lower MAX_JOBS if RAM is limited
MAX_JOBS=4 NVCC_THREADS=2 python setup.py install

# 4) Smoke test
python - <<'PY'
import torch
import flash_attn_interface
print("FA3 import ok")
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
PY
