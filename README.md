## ️ Installation

For pyradiomics integration, only works with python 3.10, so pyradiomics will be an optional dep soon
```bash
conda create -n okapy-pyradiomics python=3.10
conda activate okapy-pyradiomics

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy SimpleITK
python -m pip install pyradiomics==3.0.1 --no-build-isolation

python -m pip install -e .
```
