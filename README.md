# Prediction

Execution time prediction by Matrix Completion; currently in a prototyping phase.

Please note that our code base is optimized for cross-validation and replicate batching on relatively small datasets, and will be highly inefficient for one-off training or significantly larger datasets. In particular, we trade off memory usage and complexity (both time/space) for higher GPU utilization and overall better performance, but only when training hundreds of replicates simultaneously on datasets of similar size to ours. Training with a small number of replicates will suffer from high overhead and low GPU utilization, while training on significantly larger datasets will suffer from memory exhaustion and disproportionately higher compute usage.

## Experiments

Ablations: sparsity = [0.2, 0.4, 0.6, 0.8]
- Linear (r = 1, 2, 4, 8, 16, 32, 64, 128)
    ```
    python3 experiments.py Lr1 Lr2 Lr4 Lr8 Lr16 Lr32 Lr64 Lr128
    ```
- Embedding (r = 8, 16, 32, 64, 128)
    ```
    python3 experiments.py Er8 Er16 Er32 Er64 Er128 -s 0.2 0.4 0.6 0.8
    ```
- Embedding (q = 2, 4, 8, 16, 32)
    ```
    python3 experiments.py Eq2 Eq4 Eq8 Eq16 Eq32 -s 0.2 0.4 0.6 0.8
    ```
- Interference (s = 1, 2, 3, 4)
    ```
    python3 experiments.py Is1 Is2 Is3 Is4 -s 0.2 0.4 0.6 0.8
    ```

Full: [0.1, 0.2, ... 0.9]
- Embedding (r=64, q=4)
- Interference (r=64, q=4, s=3)
- Embedding, + interference data
- Device Only
- Module Only

```
python3 experiments.py embedding interference device_only module_only ignore
```
