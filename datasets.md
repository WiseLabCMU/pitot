Dataset conventions:
- `t: float[N]`: observed runtime (execution time).
- `i_{axis}: int[N]`: index along each axis into matrix/tensor axes.
- `d_{axis}: float[axis:len, axis:features]`: side information for each axis.
- `n_{axis}: str[axis:len]`: names corresponding to values in each axis.
- `f_{axis}: str[axis:features]`: feature names corresponding to side information for each axis.

Dataset entry names:
- `data.npz`: `workload`, `platform`; contains `d_{}`, `n_{}`, `f_{}` data/metadata.
- `if2.npz`: `workload`, `platform`, `interference0`
- `if3.npz`: `workload`, `platform`, `interference0`, `interference1`
- `if4.npz`: `workload`, `platform`, `interference0`, `interference1`, `interference2`
