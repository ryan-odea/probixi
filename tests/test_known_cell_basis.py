import math

import pytest
import torch

from probixi.indexer.indexer import Indexer
from probixi.indexer.lattice import cell_to_B
from probixi.indexer.refine import RefineResult, _axis_angle_to_rotation
from probixi.io import CellParams

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.cuda),
    pytest.param("mps", marks=pytest.mark.mps),
]


def _require(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("angles", [(89.37, 84.94, 67.84), (90.0, 110.0, 90.0)])
def test_known_cell_candidate_preserves_basis(geometry_dict, device, angles):
    _require(device)
    cell = CellParams(14.97, 18.85, 18.89, *map(math.radians, angles))
    idx = Indexer(geometry_dict, cell, device=torch.device(device))
    B = cell_to_B(cell, device=device, dtype=idx.dtype)
    # float32 B_to_cell followed by cell_to_B errs by ~1e-7; a basis mix-up by O(1e-2)
    atol = 1e-5
    U = _axis_angle_to_rotation(B.new_tensor([0.4, -0.7, 1.2]))
    A = U @ B
    hkl = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        device=device,
    )
    rr = RefineResult(
        A=A[None],
        rmsd=A.new_zeros(1),
        n_indexed=torch.tensor([6], device=device),
        soft_score=A.new_tensor([6]),
        indexed=torch.ones((1, 6), dtype=torch.bool, device=device),
        hkl=hkl[None],
        history=A.new_zeros(1),
    )
    r = idx._build_indexing_result(
        rr, frame_index=0, n_peaks=6, positions=A.new_zeros((6, 2))
    )
    assert r is not None
    assert torch.allclose(r.U @ r.B, r.A, atol=atol)
    assert torch.allclose(r.B, B, atol=atol)
    assert torch.allclose(r.U, U, atol=atol)
    assert torch.linalg.det(r.U).item() == pytest.approx(1.0, abs=atol)
    assert torch.allclose(r.A @ r.hkl.T.to(A.dtype), A @ hkl.T.to(A.dtype), atol=atol)
