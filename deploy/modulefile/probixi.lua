-- Lmod modulefile template for Probixi. Layout + setup: deploy/README.md
-- Image at $PROBIXI_ROOT/<version>/probixi.sif; install as probixi/<version>.lua

local version = "0.3.1"  -- auto-synced by format.yml
local root    = os.getenv("PROBIXI_ROOT") or "/opt/software/probixi"
local sif     = pathJoin(root, version, "probixi.sif")

whatis("Name        : probixi")
whatis("Version     : " .. version)
whatis("Description : Self-calibrating probabilistic peak finding for serial X-ray crystallography")
whatis("URL         : https://github.com/ryan-odea/probixi")

help([[
Probixi peak finder, packaged as an Apptainer container.

  probixi -i <input> ...      runs `apptainer run --nv $PROBIXI_SIF ...`

--nv exposes the host NVIDIA GPU(s). On a CPU-only node, call apptainer
directly without it:  apptainer run "$PROBIXI_SIF" -i <input> ...

Fetch the image once from GHCR (per version):
  mkdir -p "]] .. root .. [[/]] .. version .. [["
  apptainer pull "]] .. sif .. [[" \
      oras://ghcr.io/ryan-odea/probixi:]] .. version .. [[
]])

-- warn if the image hasn't been pulled
if not isFile(sif) then
    LmodMessage("probixi: image not found at " .. sif ..
                " -- pull it from GHCR (see `module help probixi`) or set PROBIXI_ROOT.")
end

setenv("PROBIXI_SIF", sif)

-- shell function so it resolves in batch scripts too
set_shell_function("probixi",
    'apptainer run --nv "$PROBIXI_SIF" "$@"',
    'apptainer run --nv "$PROBIXI_SIF" $*')
