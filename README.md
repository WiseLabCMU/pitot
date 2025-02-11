# Pitot: Runtime Prediction for Edge Computing

Code and dataset for Pitot: Runtime Prediction for Edge Computing with Conformal Matrix Completion.

![](https://tianshu.io/resources/pitot.png)

## Usage

1. **Setup**: Assuming you have a nvidia GPU, you can simply install all dependencies with conda and poetry:
    ```sh
    conda create -n pitot python=3.12
    poetry install
    ```
    - The processed dataset used in the paper is included in the `data/` folder.

2. **Train Models**: To replicate the experiments shown in the paper:
    ```sh
    make splits
    make experiments
    ```
    - `make experiments` calls `python manage.py train`, which will automatically run all experiments from the specified list which are not present in `results/`. Note that each method result folder will contain a `config.json` with all hyperparameters used.
    - See `pitot/_presets.py` for the programmatically generated list of experiments.
    - This assumes that `python` points to the environment python which `poetry install` was run inside; if this is not the case, you can also modify the makefile with `PYTHON=your/python/bin`.

3. **Evaluate**: The results are evaluated summarized to make them easier to analyze later:
    ```sh
    make evaluate -j16
    make alias
    ```
    - This runs on CPU, so can be run simultaneously while training is in progress.
    - Some runs are shared between multiple ablations, so are `alias`'d together for convenience with `make alias`.

4. **Analyze**: The plots shown in the paper can be generated with
    ```sh
    make figures 
    ```

## Dataset Structure

Dataset conventions:
- `t: float[N]`: observed runtime (execution time), in seconds.
- `i_{axis}: int[N]`: index along each axis into matrix/tensor axes.
- `d_{axis}: float[axis:len, axis:features]`: side information for each axis.
- `n_{axis}: str[axis:len]`: names corresponding to values in each axis.
- `f_{axis}: str[axis:features]`: feature names corresponding to side information for each axis.

Dataset entry names:
- `data.npz`: `workload`, `platform`; contains `d_{}`, `n_{}`, `f_{}` data/metadata.
- `if2.npz`: `workload`, `platform`, `interference0`
- `if3.npz`: `workload`, `platform`, `interference0`, `interference1`
- `if4.npz`: `workload`, `platform`, `interference0`, `interference1`, `interference2`

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
