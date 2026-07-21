# Probixi as an Apptainer module
## Install (per version)

```bash
mkdir -p /opt/software/probixi/0.3.1
apptainer pull /opt/software/probixi/0.3.1/probixi.sif \
    oras://ghcr.io/ryan-odea/probixi:0.3.1
```

Place `modulefile/probixi.lua` in your MODULEPATH as `probixi/0.3.1.lua`. Set
`PROBIXI_ROOT` if images live somewhere other than `/opt/software/probixi`.

## Use

```bash
module load probixi
probixi -i <input> -g <geometry> -p <cell> -o <output>
```

`module load` defines `probixi` as `apptainer run --nv "$PROBIXI_SIF"` (uses the
host GPU). On a CPU-only node, run without `--nv`:

```bash
apptainer run "$PROBIXI_SIF" -i <input> ...
```
