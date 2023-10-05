## Benchmarks

Commands used to run benchmarks on the cluster to run benchmarking.

Matrix:
```sh
# matrix/polybench
hc benchmark -f `hc index -d wasm/polybench` --engine iwasm-a wasmer-a-ll wasmer-a-cl wasmtime-a iwasm-i wasm3-i wasmedge-i wasmtime-j wasmer-j-sp wasmer-j-cl --repeat 50 --limit 30

# matrix/mi-cx-vn
hc benchmark -f `hc index -d wasm/mibench` `hc index -d wasm/cortex` `hc index -d wasm/vision` --engine iwasm-a wasmer-a-ll wasmer-a-cl wasmtime-a iwasm-i wasm3-i wasmedge-i wasmtime-j wasmer-j-sp wasmer-j-cl --repeat 50 --limit 30

# matrix/libsodium
hc benchmark -f `hc index -d wasm/libsodium` --engine iwasm-a wasmer-a-ll wasmer-a-cl wasmtime-a iwasm-i wasm3-i wasmedge-i wasmtime-j wasmer-j-sp wasmer-j-cl --repeat 50 --limit 30

# matrix/python
hc benchmark -f wasm/apps/python.wasm --engine iwasm-a wasmer-a-ll wasmer-a-cl wasmtime-a iwasm-i wasm3-i wasmedge-i wasmtime-j wasmer-j-sp wasmer-j-cl --repeat 50 --limit 30 --argfile benchmarks/apps/python.json
```

Interference:
```sh
# if/aot
hc benchmark -f `hc index -d wasm` --engine iwasm-a --limit 30 --interference 2
hc benchmark -f `hc index -d wasm` --engine wasmer-a-ll --limit 30 --interference 2
hc benchmark -f `hc index -d wasm` --engine wasmer-a-cl --limit 30 --interference 2
hc benchmark -f `hc index -d wasm` --engine wasmtime-a --limit 30 --interference 2

# if/interp
hc benchmark -f `hc index -d wasm` --engine iwasm-i --limit 30 --interference 2
hc benchmark -f `hc index -d wasm` --engine wasm3-i --limit 30 --interference 2
hc benchmark -f `hc index -d wasm` --engine wasmedge-i --limit 30 --interference 2

# if/jit
hc benchmark -f `hc index -d wasm` --engine wasmtime-j --limit 30 --interference 2
hc benchmark -f `hc index -d wasm` --engine wasmer-j-sp --limit 30 --interference 2
hc benchmark -f `hc index -d wasm` --engine wasmer-j-cl --limit 30 --interference 2
```
