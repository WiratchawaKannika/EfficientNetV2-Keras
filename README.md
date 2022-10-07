# EfficientNetV2-Keras

## Dependencies
* Ubuntu 16.04 or higher (64-bit)
* Cuda <= 11.6
* python >= 3.9+
* TensorFlow >= 2.10
* tensorflow-gpu >= 2.10

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
- conda list | grep cuda
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
```
* *EfficientNetV2 models rewritten in Keras functional API From Repository.
- 🐥https://github.com/sebastian-sz/efficientnet-v2-keras.git

### 4. Verify install
*Verify the GPU setup:
```
- import tensorflow as tf
- print(tf.__version__)
```

```
- physical_devices = tf.config.list_physical_devices('GPU') 
- print("Num GPUs:", len(physical_devices))
```

``` 
- print('Num GPUs Available:', len(tf.config.experimental.list_physical_devices('GPU')))
```

```
- tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

'2022-09-29 15:09:18.107884: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /device:GPU:0 with 9454 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:17:00.0, compute capability: 7.5
2022-09-29 15:09:18.108402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /device:GPU:1 with 9629 MB memory:  -> device: 1, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:65:00.0, compute capability: 7.5
True'
```

```
- python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If a list of GPU devices is returned, you've installed TensorFlow successfully.

--------------------


#### ⛄Install TensorFlow 2.10.x with pip , GPU (Set environment. "Python 3.9+")

- https://www.tensorflow.org/install/pip







