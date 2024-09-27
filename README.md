# 2024 Andes Award - Build AI Model

The repository is for compiling the AI model based on TVM in the 2024 Andes Award competition.

## Setup ENV

1. Use `miniconda` to manage python dependencies
    ```bash
    # make sure to start with a fresh environment
    conda env remove -n tvm-build-venv
    # create the conda environment with build dependency
    conda create -n tvm-build-venv -c conda-forge \
        "llvmdev=15" \
        "cmake>=3.24" \
        git \
        python=3.11
    # enter the build environment
    conda activate tvm-build-venv
    ```
