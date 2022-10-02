# Prediction

Execution time prediction by Matrix Completion; currently in a prototyping phase.


## Experiments

Ablations: sparsity = [0.2, 0.4, 0.6, 0.8]
- Linear (r = 8, 16, 32, 64, 128)
- Embedding (r = 8, 16, 32, 64, 128)
- Embedding (u = 2, 4, 8, 16, 32)
- Interference (s = 1, 2, 4, 8, 16)
- 5x4x4

Full: [0.1, 0.2, ... 0.9]
- Linear (r=64)
- Embedding (r=64, q=?)
- Interference (r=64, q=? s=?)
- Embedding, + interference data
- 4x9
