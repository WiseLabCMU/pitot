# Prediction

Execution time prediction by Matrix Completion; currently in a prototyping phase.

Please note that our code base is optimized for cross-validation and replicate batching on relatively small datasets, and will be highly inefficient for one-off training or significantly larger datasets. In particular, we trade off memory usage and complexity (both time/space) for higher GPU utilization and overall better performance, but only when training hundreds of replicates simultaneously on datasets of similar size to ours. Training with a small number of replicates will suffer from high overhead and low GPU utilization, while training on significantly larger datasets will suffer from memory exhaustion and disproportionately higher compute usage.

## Experiments

Experiments:
```
python manage.py experiments embedding/*
python manage.py experiments features/*
python manage.py experiments linear/*
python manage.py experiments paragon/*
python manage.py experiments interference/* -d data/data.if.npz
python manage.py experiments interference3/* -d data/data.if3.npz
```
