from __future__ import annotations

import torch

from probixi.probixi import shadow_mask


def _field(value=40.0, shape=(300, 200)):
    radial = torch.full(shape, value)
    pixel = radial + torch.randn(shape) * 2.0
    return pixel, radial, torch.ones(shape, dtype=torch.bool)


def test_flat_field_has_no_shadow():
    pixel, radial, valid = _field()
    assert shadow_mask(pixel, radial, valid, grow=9) is None


def test_dark_rectangle_is_masked_and_grown():
    pixel, radial, valid = _field()
    pixel[100:200, 120:200] = torch.randn(100, 80) * 0.5  # ~0 ADU under a 40 ADU field
    m = shadow_mask(pixel, radial, valid, grow=9)
    assert m is not None
    # the rectangle itself and a 9 px margin around it are masked ...
    assert bool(m[100:200, 120:200].all())
    assert bool(m[95, 150]) and bool(m[204, 150]) and bool(m[150, 112])
    # ... and nothing far from it
    assert not bool(m[:80].any()) and not bool(m[:, :100].any())


def test_isolated_dark_pixels_are_not_a_shadow():
    pixel, radial, valid = _field()
    idx = torch.randint(0, 300, (40,)), torch.randint(0, 200, (40,))
    pixel[idx] = 0.0
    assert shadow_mask(pixel, radial, valid, grow=9) is None


def test_dead_pixels_do_not_count_as_shadow():
    pixel, radial, valid = _field()
    valid[100:200, 120:200] = False
    pixel[100:200, 120:200] = 0.0
    assert shadow_mask(pixel, radial, valid, grow=9) is None
