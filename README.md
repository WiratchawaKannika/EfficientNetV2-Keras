# EfficientNetV2-Keras

## Dependencies
* Ubuntu 16.04 or higher (64-bit)
* python >= 3.9+
* TensorFlow >= 2.10
* tensorflow-gpu >= 2.10.

### 1. Create a conda environment
```
- conda create --name tf python=3.9
- conda deactivate
- conda activate tf
- pip install --user ipykernel
- python -m ipykernel install --user --name=tf
```

### 2. GPU setup
```
- nvidia-smi
- conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
- mkdir -p $CONDA_PREFIX/etc/conda/activate.d
- echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### 3.  Install TensorFlow and Package
```
- pip install --upgrade pip
- pip install tensorflow
- pip install tensorflow-gpu
- pip install keras
- *pip install -q git+https://github.com/sebastian-sz/efficientnet-v2-keras@main
- conda install jupyter
- conda install pandas
- pip install -U scikit-image


# ⛄Install TensorFlow 2.10.x with pip , GPU (Set environment. "Python 3.9+")

- https://www.tensorflow.org/install/pip







