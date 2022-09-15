#!/bin/bash

PB="data/cfs/cfs.$1/pb.$1"
MI="data/cfs/cfs.$1/mi.$1"
CX="data/cfs/cfs.$1/cx.$1"
VN="data/cfs/cfs.$1/vn.$1"

MAT="python3 forecast.py matrix --path"

$MAT $PB
$MAT $MI
$MAT $CX
$MAT $VN

python3 forecast.py join \
    --datasets $PB.npz $MI.npz $CX.npz $VN.npz \
    --opcodes data/opcodes.npz \
    --runtimes data/runtimes.npz \
    --out data/cfs/cfs.$1.npz
