from types import SimpleNamespace

import torch

from probixi import Probixi
from probixi.indexer import FrameIndexResult, FrameIndexStream, IndexStats


def test_calibration_boundaries_ignore_batches_and_worker_start(monkeypatch):
    def run(start, batch_size):
        p = object.__new__(Probixi)
        p._frame_scales = {}
        p._calibration_options = {"n_seed": 2}
        p._calibration_boundary = 0
        p.threshold_calibration = SimpleNamespace(threshold=5.0)
        calls, thresholds = [], []

        def calibrate(boundary):
            calls.append(boundary)
            p._calibration_boundary = boundary
            p.threshold_calibration = SimpleNamespace(threshold=5.0 + boundary)

        def peaks(frames, start_index, update_noise):
            assert not update_noise
            return enumerate(frames, start_index)

        def index(peaks, batch_size, bright_threshold, enrich_alpha):
            stats = IndexStats()

            def generate():
                for i, _ in peaks:
                    thresholds.append((i, bright_threshold))
                    stats.frames += 1
                    yield FrameIndexResult(i, [], torch.empty((0, 2)), torch.empty(0))

            return FrameIndexStream(generate(), stats)

        monkeypatch.setattr(p, "_recalibrate", calibrate)
        monkeypatch.setattr(p, "peak_stream", peaks)
        p.indexer = SimpleNamespace(index_frame_stream=index)
        stream = p.index_frame_stream(
            (torch.zeros((1, 1)) for _ in range(start, 11)),
            start_index=start,
            batch_size=batch_size,
            recalibrate_every=4,
        )
        assert [r.frame_index for r in stream] == list(range(start, 11))
        assert stream.stats.frames == 11 - start
        return calls, thresholds

    calls, thresholds = run(0, 3)
    assert calls == [4, 8]
    assert thresholds == [(i, 5.0 + i // 4 * 4) for i in range(11)]
    assert run(0, 7) == (calls, thresholds)
    assert run(5, 3) == ([4, 8], thresholds[5:])


def test_fresh_calibration_keeps_scale_reference(monkeypatch):
    p = object.__new__(Probixi)
    p._calibration_options = {"n_seed": 2, "target_noise_peaks": 5}
    old_noise, reference = object(), object()
    p._noise = old_noise
    p._finder = object()
    p._scale_ref = reference
    p.device = torch.device("cpu")
    requested = []

    def frames(start, stop):
        requested.append((start, stop))
        return iter(range(start, stop))

    def calibrate(seed_frames, **options):
        assert p._noise is None and p._finder is None
        assert list(seed_frames) == [8, 9]
        p._noise = object()
        p._scale_ref = object()
        p.threshold_calibration = SimpleNamespace(threshold=6.0)

    monkeypatch.setattr(p, "frames", frames)
    monkeypatch.setattr(p, "calibrate", calibrate)
    p._recalibrate(8)
    assert requested == [(8, 10)]
    assert p._noise is not old_noise
    assert p._scale_ref is reference
    assert p._calibration_boundary == 8
    assert p.threshold_calibration.threshold == 6.0


def test_sampled_frames_are_deterministic_spread_and_disjoint():
    def pipeline(seed, n_frames=1000):
        p = object.__new__(Probixi)
        p.random_seed = seed
        p.loader = SimpleNamespace(metadata=SimpleNamespace(n_frames=n_frames))
        return p

    p = pipeline(1988)
    warmup = p._sample_frame_indices(32)
    assert len(warmup) == 32 and warmup == sorted(warmup)
    assert warmup == p._sample_frame_indices(32)
    assert warmup != pipeline(1989)._sample_frame_indices(32)
    # spread over the run, not the leading frames
    assert max(warmup) > 500
    training = p._sample_frame_indices(32, warmup)
    assert not set(training) & set(warmup)
    # a run shorter than the request yields every frame
    assert pipeline(1988, 8)._sample_frame_indices(32) == list(range(8))
