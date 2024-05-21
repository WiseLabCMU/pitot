# Pitot

## Setup

Our experiments were run on Python 3.11, though any Python version supporting type checking by beartype and jaxtyping (3.7+) should work.

**NOTE**: assuming you want to use a GPU, you will need to [install JAX](https://github.com/google/jax#installation) with a version that matches your CUDA and CuDNN version. Since JAX can't be installed automatically with GPU support, it is not included in `requirements.txt`.

```sh
pip install -r requirements.txt
```

If you would like to use the exact versions of each dependency that we used, please use
```sh
pip install -r requirements-pinned.txt
```

Our environment runs nvidia driver `535.113.01` and CUDA `11.8.89`.

## Repository Structure

Python modules:
- `prediction`: core *resuable* library runtime prediction functionality.
- `pitot`: implementation of *pitot* and other baselines shown in our paper.
- Core python code is type-annotated wherever possible, and can be statically typechecked with mypy.

Scripts:
- `preprocess.py` / `preprocess`: data preprocessing steps to turn raw data into `.npz` dataset files.
- `manage.py` / `scripts`: main split, training, and evaluation scripts.
- `plot.py` / `plot`: scripts for drawing the figures shown in the paper.

Data files:
- `data`: after unzipping `data.tar`, the resulting `data` folder should be placed in the repository directory.
- `splits`: this directory is created by `make splits`. To make sure that you use the same data splits (even with RNG changes or other code changes), you can download the supplied `splits.tar` file and extract it here.

Result files:
- `results`: outputs of each experiment are saved here. Has the following structure:
    ```sh
    results/
        method/path/     # can be an arbitrary number of levels
            config.json  # configuration parameters used for training
            0.1/         # data split size
                0.npz    # training log for replicate 0
                0.pkl    # weights for replicate 0
                ...
            ...
        ...
    ```
- `summary`: output of method evaluation.
    - Each `results/method/path/` is turned into one `summary/method/path.npz`.
    - Entries are stacked with data split size as the first axis and the replicate as the second axis (on top of any remaining axes, e.g. target quantile)

## Experiments

To replicate the experiments shown in the paper:
```sh
make experiments
```
- Should take ~5 hours on a RTX 4090 / 7950X; your mileage may vary.

```sh
make evaluate -j16
```
- Change `-j` based on your CPU; should take a few minutes.
- Each evaluation runs CPU-only (evaluation is mostly data marshalling), so the only practical limit on concurrency is total memory.

```sh
make alias
```
- To save training time, duplicated entries are provided logically via symlinks.

```sh
make figures
```
- See `figures/` for outputs.
