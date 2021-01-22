# 3D object classification

## Descriptions

This repository contains code for [demo app](https://github.com/KernelA/made-ml-demo-app)


## Requirements

1. Python 3.7 or higher.
2. Anaconda
3. CUDA 10.2 or higher. You can use also CUDA 10.1 and 10.0 but file with environment require modification of dependencies.
4. Git LFS

## How to run

Create new conda environment:
```
conda env create -n env_name --file ./conda-env.yml
conda activate env_name
```

This project use DVC as data version system. You can see all pipeline:
```
dvc dag
```

For model training:
```
dvc repro export_model
```
*Note, full reproducibility is not guaranteed.*

Final weights are stored in Git LFS.
