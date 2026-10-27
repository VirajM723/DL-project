"""Unit / integration tests for TrackNetV5_TinyObject_Tracking.ipynb.

Run:  .venv/Scripts/python -m pytest tests -q            (all, ~3-4 min on a laptop CPU)
      .venv/Scripts/python -m pytest tests -q -m "not slow"   (unit tests only, ~1 min)
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import NOTEBOOK


# ----------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------
def test_profiles_are_shape_compatible(nb):
    """3 max-pools need /8; the R-STR patch sizes (8 full, 16 slim) must tile the frame."""
    for name, p in nb.PROFILES.items():
        for d in (8, 16):
            assert p["W"] % d == 0 and p["H"] % d == 0, (name, d)
        assert p["clip_len"] >= 3


def test_tolerance_scales_with_resolution(nb):
    assert nb.SC == pytest.approx(nb.W / 288)
    assert nb.CFG["tol"] == pytest.approx(3.0 * nb.SC, abs=0.01)
    assert nb.CFG["disk_r"] == pytest.approx(30 * nb.W / 1280)


# ----------------------------------------------------------------------------------------------
# synthetic data generator
# ----------------------------------------------------------------------------------------------
def _ball_pixels(nb, seed, L=60):
    """Render the same clip twice with two ball colours: pixels that differ are exactly the ball."""
    fa, la = nb.render_clip(seed, L, nb.W, nb.H, ball_color=(1.0, 0.0, 1.0))
    fb, lb = nb.render_clip(seed, L, nb.W, nb.H, ball_color=(0.0, 1.0, 0.0))
    np.testing.assert_array_equal(la, lb)                      # labels don't depend on the colour
    diff = np.abs(fa.astype(int) - fb.astype(int)).max(-1)     # [L,H,W]
    return la, diff > 25


def test_render_is_deterministic(nb):
    f1, l1 = nb.render_clip(7, 12, nb.W, nb.H)
    f2, l2 = nb.render_clip(7, 12, nb.W, nb.H)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(l1, l2)


def test_labels_match_rendered_ball(nb):
    n_vis = n_occ = n_dead = 0
    occ_hidden, occ_visible_share = [], []
    for seed in range(300, 316):
        lab, ball = _ball_pixels(nb, seed)
        for t in range(len(lab)):
            x, y, vis = lab[t, :3]
            ys, xs = np.nonzero(ball[t])
            if vis == 1:
                n_vis += 1
                assert len(xs) > 0, f"seed {seed} t {t}: visible label but no ball pixels"
                d = np.hypot(xs - x, ys - y).min()
                assert d <= 1.5, f"seed {seed} t {t}: nearest ball pixel {d:.2f}px from label"
            elif vis == 2:
                n_occ += 1
                xi, yi = int(round(x)), int(round(y))
                occ_hidden.append(not (0 <= yi < nb.H and 0 <= xi < nb.W and ball[t, yi, xi]))
                r = lab[t, 4]
                occ_visible_share.append(np.sum(np.hypot(xs - x, ys - y) <= r) / (np.pi * r * r))
            elif 0 <= x < nb.W and 0 <= y < nb.H:              # vis 0 while inside the frame => dead ball
                n_dead += 1
                assert len(xs) == 0, f"seed {seed} t {t}: dead-ball frame shows the ball"
            # (centre out of frame: a blur streak may still reach inside the border, as in TrackNet labels)
    assert n_vis > 300 and n_dead > 0
    # regression: a 1-px racket string over the ball centre (or a covered blur tail) used to count as occlusion
    if n_occ:
        assert np.mean(occ_hidden) >= 0.75, f"only {np.mean(occ_hidden):.0%} of occluded labels are hidden"
        assert np.median(occ_visible_share) <= 0.35, f"occluded balls are mostly visible ({np.median(occ_visible_share):.2f})"


def test_label_statistics(nb):
    labs = np.concatenate([nb.render_clip(s, 60, nb.W, nb.H)[1] for s in range(500, 512)])
    vis = labs[:, 2]
    assert 0.01 <= np.mean(vis == 2) <= 0.2, "occlusion rate out of the intended range"
    assert 0.03 <= np.mean(vis == 0) <= 0.3, "ball-absent rate out of the intended range"
    pres = vis > 0
    assert labs[pres, 4].min() > 0.8 * nb.SC and labs[pres, 4].max() < 4.0 * nb.SC   # tiny ball
    assert np.percentile(labs[pres, 3], 95) > 8 * nb.SC                                 # fast ball


def test_camera_shake_moves_whole_frame(nb):
    """Regression: jitter used to move only ball + players, leaving the court static."""
    fr, _ = nb.render_clip(11, 60, nb.W, nb.H)
    top = int(nb.H * 0.12)                                     # crowd band: static scene content
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)[:top].astype(np.float32) for f in fr]
    shifts = [np.hypot(*cv2.phaseCorrelate(g[0], g[t])[0]) for t in range(5, 60, 5)]
    assert max(shifts) > 0.3, f"background never moves (max shift {max(shifts):.2f}px)"
    assert max(shifts) < 9 * nb.H / 160                       # |jitter| <= 3 px per axis (at 160 px height)


# ----------------------------------------------------------------------------------------------
# targets + dataset
# ----------------------------------------------------------------------------------------------
def _brute_heatmap(nb, lab, kind):
    out = np.zeros((len(lab), nb.H, nb.W), np.float32)
    for i, (x, y, vis, _, r, _) in enumerate(lab):
        if vis == 0:
            continue
        d2 = (nb.XX - x) ** 2 + (nb.YY - y) ** 2
        out[i] = (d2 <= nb.CFG["disk_r"] ** 2) if kind == "disk" else np.exp(-d2 / (2 * max(1.0, 0.8 * r) ** 2))
    return out


@pytest.mark.parametrize("kind", ["disk", "gauss"])
def test_local_heatmap_equals_full_frame(nb, kind):
    rng = np.random.default_rng(0)
    lab = np.zeros((40, 6), np.float32)
    lab[:, 0] = rng.uniform(-3, nb.W + 3, 40); lab[:, 1] = rng.uniform(-3, nb.H + 3, 40)
    lab[:, 2] = rng.choice([0, 1, 2], 40); lab[:, 4] = rng.uniform(1, 4, 40)
    np.testing.assert_allclose(nb.make_heatmap(lab, kind), _brute_heatmap(nb, lab, kind), atol=5e-4)


def test_gaussian_target_peaks_at_label(nb):
    lab = np.array([[100.3, 50.7, 1, 0, 2.5, 0]], np.float32)
    hm = nb.make_heatmap(lab, "gauss")[0]
    iy, ix = np.unravel_index(hm.argmax(), hm.shape)
    assert (ix, iy) == (100, 51)
    assert nb.make_heatmap(np.array([[100, 50, 0, 0, 2, 0]], np.float32), "gauss").max() == 0


def test_dataset_flip_is_consistent(nb, monkeypatch):
    ds_plain = nb.WindowDataset(nb.DATA["train"], "gauss", augment=False)
    ds_aug = nb.WindowDataset(nb.DATA["train"], "gauss", augment=True)
    monkeypatch.setattr(nb.random, "random", lambda: 0.0)                  # always flip
    monkeypatch.setattr(nb.random, "uniform", lambda a, b: (a + b) / 2)    # identity brightness/contrast
    i = next(k for k in range(len(ds_plain)) if ds_plain[k][2][:, 2].min() > 0)
    x0, y0, l0 = ds_plain[i]
    x1, y1, l1 = ds_aug[i]
    torch.testing.assert_close(x1, x0.flip(-1))
    torch.testing.assert_close(y1, y0.flip(-1), atol=5e-4, rtol=0)
    torch.testing.assert_close(l1[:, 0], (nb.W - 1) - l0[:, 0])


def test_dataset_channel_layout(nb):
    """9 input channels = [f0 RGB, f1 RGB, f2 RGB]; MIMO targets align with the 3 frames."""
    ds = nb.WindowDataset(nb.DATA["train"], "gauss")
    c, s = ds.idx[5]
    x, y, lab = ds[5]
    fr = nb.DATA["train"]["frames"][c, s:s + 3].astype(np.float32) / 255
    for k in range(3):
        np.testing.assert_allclose(x[3 * k:3 * k + 3].numpy(), fr[k].transpose(2, 0, 1))
    np.testing.assert_array_equal(lab.numpy(), nb.DATA["train"]["labels"][c, s:s + 3])


# ----------------------------------------------------------------------------------------------
# inference windowing
# ----------------------------------------------------------------------------------------------
class _RedProbe(nn.Module):
    """Fake model whose output heatmap for each frame equals that frame's red channel."""
    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W

    def forward(self, x):
        red = x.view(x.shape[0], 3, 3, self.H, self.W)[:, :, 0]
        return torch.logit(red.clamp(1e-4, 1 - 1e-4))


@pytest.mark.parametrize("clip_len", [60, 61, 62, 3])
def test_predict_heatmaps_covers_and_aligns_every_frame(nb, clip_len):
    rng = np.random.default_rng(clip_len)
    split = dict(frames=[rng.integers(0, 256, (clip_len, nb.H, nb.W, 3), dtype=np.uint8) for _ in range(2)])
    hms = nb.predict_heatmaps(_RedProbe(nb.H, nb.W), split, batch=4)
    for hm, fr in zip(hms, split["frames"]):
        assert hm.shape == (clip_len, nb.H, nb.W)
        assert np.abs(hm.astype(int) - fr[..., 0].astype(int)).max() <= 1


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def test_all_variants_forward_backward(nb):
    x = torch.rand(2, 9, nb.H, nb.W)
    params = {}
    for name, spec in nb.VARIANTS.items():
        m = nb.TrackNet(**spec).train()
        out = m(x)
        assert out.shape == (2, 3, nb.H, nb.W) and torch.isfinite(out).all(), name
        nb.wbce_loss(out, torch.zeros_like(out)).backward()
        first_conv = m.backbone.e1[0].weight
        assert first_conv.grad is not None and first_conv.grad.abs().sum() > 0, name
        params[name] = sum(p.numel() for p in m.parameters())
    assert params["V5 (faithful)"] > params["V2 (baseline)"]
    assert params["V5 +M1+M2+M3"] < params["V5 +M1+M2"]                      # slim head
    assert params["V5-Lite+ (M1–M4)"] < 0.3 * params["V5 +M1+M2+M3"]         # depthwise backbone


def test_input_channels(nb):
    assert nb.TrackNet(mdd=None).backbone.e1[0].in_channels == 9
    assert nb.TrackNet(mdd="v5").backbone.e1[0].in_channels == 13             # V5: 9 RGB + 2x2 polarity maps
    assert nb.TrackNet(mdd="m2").backbone.e1[0].in_channels == 15             # M2: + 2 direction channels


def _moving_dot(nb, positions, val=1.0):
    x = torch.zeros(1, 9, nb.H, nb.W)
    for k, (px, py) in enumerate(positions):
        x[0, 3 * k:3 * k + 3, py - 1:py + 2, px - 1:px + 2] = val
    return x


def test_mdd_polarity_semantics(nb):
    """P+ fires where the ball arrives, P- where it leaves; the M2 direction channel is their signed difference."""
    pos = [(40, 40), (60, 40), (80, 40)]
    x = _moving_dot(nb, pos)
    for per_channel in (False, True):
        mdd = nb.MDD(per_channel=per_channel, direction=per_channel)
        with torch.no_grad():
            A = mdd(x)[0]                                                    # [pair, polarity, H, W]
        assert A.shape == (2, 2, nb.H, nb.W) and (A >= 0).all() and (A <= 1).all()
        bg = A[0, 0, 5, 5].item()
        assert A[0, 0, 40, 60] > bg + 0.5 and A[0, 1, 40, 40] > bg + 0.5     # pair 0: arrive @60, leave @40
        assert A[1, 0, 40, 80] > bg + 0.5 and A[1, 1, 40, 60] > bg + 0.5     # pair 1: arrive @80, leave @60
        assert abs(A[0, 0, 40, 40] - bg) < 1e-4                              # no positive response where it left
    mdd = nb.MDD(per_channel=True, direction=True)
    with torch.no_grad():
        inp = mdd.build_input(x, mdd(x))
    assert inp.shape[1] == 15
    torch.testing.assert_close(inp[:, 0:3], x[:, 0:3])                       # f0
    torch.testing.assert_close(inp[:, 6:9], x[:, 3:6])                       # f1 after [A1+, A1-, dir1]
    torch.testing.assert_close(inp[:, 12:15], x[:, 6:9])                     # f2
    assert inp[0, 5, 40, 60] > 0.5 and inp[0, 5, 40, 40] < -0.5             # direction dipole


def test_mdd_parameterisation(nb):
    k, m = nb.MDD(per_channel=False).k_m()
    assert k.shape == (1, 1, 1, 1, 1)
    k, m = nb.MDD(per_channel=True).k_m()
    assert k.shape == (1, 2, 2, 1, 1) and (k > 0).all() and (m.abs() <= 0.6).all()
    assert len(list(nb.MDD(per_channel=True).parameters())) == 2 and nb.MDD(per_channel=True).alpha.numel() == 4


def test_rstr_head_starts_as_identity(nb):
    x = torch.rand(1, 9, nb.H, nb.W)
    for name in ("V5 (faithful)", "V5 +M1+M2+M3"):
        m = nb.TrackNet(**nb.VARIANTS[name]).eval()
        with torch.no_grad():
            out, parts = m(x, return_parts=True)
        torch.testing.assert_close(parts["delta"], torch.zeros_like(parts["delta"]))
        torch.testing.assert_close(out, parts["draft"] + m.gate_w * parts["gate"])


def test_faithful_v5_train_inference_mismatch_and_m3_fix(nb):
    """Faithful V5 trains the head on Dropout(Draft) and never trains the MDD fusion; M3 does."""
    x = torch.rand(2, 9, nb.H, nb.W)
    for name, expect_grad in (("V5 (faithful)", False), ("V5 +M1+M2+M3", True)):
        torch.manual_seed(0)
        m = nb.TrackNet(**nb.VARIANTS[name]).train()
        m(x).sum().backward()
        assert (m.gate_w.grad is not None) == expect_grad, name


@pytest.mark.parametrize("patch,dim,depth,heads", [(8, 64, 2, 4), (16, 32, 1, 2)])
def test_tsatt_head_at_gpu_resolution(nb, patch, dim, depth, heads):
    head = nb.TSATTHead(288, 512, patch=patch, dim=dim, depth=depth, heads=heads)
    nn.init.normal_(head.proj.weight, std=0.01)
    d = torch.randn(1, 3, 288, 512)
    with torch.no_grad():
        out = head(d, torch.rand_like(d))
    assert out.shape == d.shape and torch.isfinite(out).all() and out.abs().sum() > 0


def test_tsatt_is_spatio_temporal(nb):
    """Changing one frame must influence the residual of the other frames (temporal attention works)."""
    torch.manual_seed(0)
    head = nb.TSATTHead(nb.H, nb.W, **nb.RSTR_CFG["slim"]).eval()
    nn.init.normal_(head.proj.weight, std=0.1)
    d = torch.randn(1, 3, nb.H, nb.W)
    d2 = d.clone(); d2[0, 2] += 3.0
    with torch.no_grad():
        a, b = head(d, torch.zeros_like(d)), head(d2, torch.zeros_like(d))
    assert (a[0, 0] - b[0, 0]).abs().max() > 1e-4


# ----------------------------------------------------------------------------------------------
# loss, decoding, metrics
# ----------------------------------------------------------------------------------------------
def test_wbce_matches_formula_and_is_stable(nb):
    torch.manual_seed(0)
    z = torch.randn(4, 3, 8, 8); y = torch.rand_like(z)
    p = torch.sigmoid(z)
    ref = -((1 - p) ** 2 * y * torch.log(p) + p ** 2 * (1 - y) * torch.log(1 - p)).mean()
    torch.testing.assert_close(nb.wbce_loss(z, y), ref)
    z = torch.tensor([[-80.0, 80.0]], requires_grad=True)
    loss = nb.wbce_loss(z, torch.tensor([[1.0, 0.0]]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def _gauss(nb, x, y, sig):
    return np.exp(-((nb.XX - x) ** 2 + (nb.YY - y) ** 2) / (2 * sig ** 2)).astype(np.float32)


@pytest.mark.parametrize("radius", [1.3, 2.6, 3.3])                    # smallest / typical / largest ball (x SC)
def test_softargmax_is_unbiased_subpixel(nb, radius):
    """Regression: a plain centroid in a 5x5 window around the integer peak was biased by ~0.24 px."""
    rng = np.random.default_rng(1)
    sig = max(1.0, 0.8 * radius * nb.SC)
    new, old = [], []
    for _ in range(60):
        x, y = rng.uniform(25, nb.W - 25), rng.uniform(25, nb.H - 25)
        hm = _gauss(nb, x, y, sig)
        xy, _ = nb.decode_softargmax(hm, 0.5)
        new.append(math.hypot(xy[0] - x, xy[1] - y))
        xy, _ = nb.decode_softargmax(hm, 0.5, k=2, rel=0.0)               # the previous decoder
        old.append(math.hypot(xy[0] - x, xy[1] - y))
    assert np.mean(new) < 0.05, f"sigma {sig:.2f}: mean error {np.mean(new):.3f}px"
    assert np.mean(new) <= np.mean(old)


def test_softargmax_robust_to_background(nb):
    """Low-level background activations (e.g. on court lines) must not drag the estimate."""
    rng = np.random.default_rng(4)
    hm = _gauss(nb, 100.3, 60.6, 2.0) + rng.uniform(0, 0.05, (nb.H, nb.W)).astype(np.float32)
    xy, _ = nb.decode_softargmax(hm, 0.5)
    assert math.hypot(xy[0] - 100.3, xy[1] - 60.6) < 0.15


def test_contour_decoder(nb):
    x, y = 120.4, 70.6
    disk = ((nb.XX - x) ** 2 + (nb.YY - y) ** 2 <= nb.CFG["disk_r"] ** 2).astype(np.float32)
    xy, conf = nb.decode_contour(disk, 0.5)
    assert math.hypot(xy[0] - x, xy[1] - y) <= 0.75 and conf == 1.0
    two = np.maximum(disk, ((nb.XX - 30) ** 2 + (nb.YY - 30) ** 2 <= 4).astype(np.float32) * 0.9)
    xy2, _ = nb.decode_contour(two, 0.5)
    assert math.hypot(xy2[0] - x, xy2[1] - y) <= 0.75                  # largest blob wins


@pytest.mark.parametrize("dec", ["contour", "softargmax"])
def test_decoders_respect_threshold(nb, dec):
    hm = _gauss(nb, 50, 50, 2.0) * 0.4
    xy, conf = nb.DECODERS[dec](hm, 0.5)
    assert xy is None


def test_evaluate_counts_and_metrics(nb):
    nan = np.nan
    #           present & hit | present & far | absent & pred | present & none | absent & none
    labels = [np.array([[10, 10, 1, 0, 2, 0], [10, 10, 2, 0, 2, 0], [0, 0, 0, 0, 2, 0],
                        [10, 10, 1, 0, 2, 0], [0, 0, 0, 0, 2, 0]], np.float32)]
    preds = [np.array([[11, 10], [30, 30], [5, 5], [nan, nan], [nan, nan]], np.float32)]
    r = nb.evaluate(preds, labels, tol=3.0)
    assert (r["TP"], r["FP1"], r["FP2"], r["FN"], r["TN"]) == (1, 1, 1, 1, 1)
    assert r["prec"] == pytest.approx(1 / 3) and r["rec"] == pytest.approx(1 / 2)
    assert r["f1"] == pytest.approx(0.4) and r["acc"] == pytest.approx(0.4)
    assert r["err_mean"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------
# M5: Kalman gating + ballistic gap filling
# ----------------------------------------------------------------------------------------------
def _parabola(n=40, x0=30.0, y0=120.0, vx=5.0, vy=-9.0, g=0.5):
    t = np.arange(n, dtype=np.float32)
    return np.stack([x0 + vx * t, y0 + vy * t + 0.5 * g * t ** 2], 1).astype(np.float32)


def test_kalman_keeps_clean_track(nb):
    rng = np.random.default_rng(0)
    gt = _parabola()
    P = gt + rng.normal(0, 0.3, gt.shape).astype(np.float32)
    out = nb.kalman_refine(P, np.ones(len(P)))
    assert np.mean(~np.isnan(out[:, 0])) >= 0.95
    ok = ~np.isnan(out[:, 0])
    np.testing.assert_allclose(out[ok], P[ok])


def test_kalman_rejects_outlier_and_fills_gaps(nb):
    rng = np.random.default_rng(1)
    gt = _parabola()
    P = gt + rng.normal(0, 0.3, gt.shape).astype(np.float32)
    P[20] = gt[20] + np.array([45.0, -40.0])                           # isolated false detection
    P[10:13] = np.nan                                                  # 3-frame occlusion
    out = nb.kalman_refine(P, np.ones(len(P)))
    assert np.hypot(*(out[20] - gt[20])) < 2.0, "outlier was not replaced by the ballistic fill"
    assert np.max(np.hypot(*(out[10:13] - gt[10:13]).T)) < 1.5, "occlusion gap badly filled"


def test_kalman_reinitialises_after_hit(nb):
    rng = np.random.default_rng(2)
    a = _parabola(20)
    b = _parabola(20, x0=a[-1, 0], y0=a[-1, 1], vx=-6.0, vy=-8.0)     # racket hit: velocity reverses
    P = np.concatenate([a, b]) + rng.normal(0, 0.3, (40, 2)).astype(np.float32)
    out = nb.kalman_refine(P, np.ones(len(P)))
    assert np.mean(~np.isnan(out[:, 0])) >= 0.9


def test_kalman_never_fills_outside_frame(nb):
    """A lob that leaves through the top edge for 5 frames: the gap is bounded by detections but must stay empty."""
    gt = _parabola(20, x0=100.0, y0=2.0, vx=3.0, vy=-3.0, g=1.0)     # y < 0 for t = 1..5
    P = gt.copy()
    P[gt[:, 1] < 0] = np.nan
    assert np.isnan(P[1:6, 0]).all() and not np.isnan(P[6:, 0]).any()
    out = nb.kalman_refine(P, np.ones(len(P)), max_gap=6)
    assert np.isnan(out[1:6, 0]).all(), out[1:6]
    filled = out[~np.isnan(out[:, 0])]
    assert ((filled[:, 1] >= 0) & (filled[:, 1] < nb.H)).all()


def test_m5_improves_noisy_detections_on_rendered_trajectories(nb):
    """End-to-end check of M5 on realistic rallies (bounces, hits, occlusion) with simulated detector errors."""
    rng = np.random.default_rng(3)
    labels, raw, conf = [], [], []
    for s in range(700, 716):
        lab = nb.render_clip(s, 60, nb.W, nb.H)[1]
        P = np.full((60, 2), np.nan, np.float32)
        pres = lab[:, 2] > 0
        P[pres] = lab[pres, :2] + rng.normal(0, 0.5 * nb.SC, (pres.sum(), 2))
        P[lab[:, 2] == 2] = np.nan                                     # detector misses occluded balls
        P[rng.random(60) < 0.08] = np.nan                              # random misses
        out = rng.random(60) < 0.06                                    # false positives anywhere
        P[out] = np.stack([rng.uniform(0, nb.W, out.sum()), rng.uniform(0, nb.H, out.sum())], 1)
        labels.append(lab); raw.append(P); conf.append(np.where(np.isnan(P[:, 0]), 0, 0.9))
    before = nb.evaluate(raw, labels)
    after = nb.evaluate([nb.kalman_refine(p, c) for p, c in zip(raw, conf)], labels)
    assert after["prec"] > before["prec"] + 0.02, (before, after)
    assert after["f1"] > before["f1"], (before, after)


# ----------------------------------------------------------------------------------------------
# real-data loader + bookkeeping
# ----------------------------------------------------------------------------------------------
def test_real_tennis_loader(nb, tmp_path, monkeypatch):
    clip = tmp_path / "Dataset" / "game1" / "Clip1"
    clip.mkdir(parents=True)
    rows = [("0000.jpg", 1, 640, 360), ("0001.jpg", 0, "", ""), ("0002.jpg", 3, 100, 200),
            ("0003.jpg", 2, 1279, 719), ("0004.jpg", 1, 0, 0), ("0005.jpg", 1, 320, 180), ("0006.jpg", 1, 5, 5)]
    for name, *_ in rows:
        cv2.imwrite(str(clip / name), np.full((720, 1280, 3), 128, np.uint8))
    (clip / "Label.csv").write_text("file name,visibility,x-coordinate,y-coordinate,status\n" +
                                    "\n".join(f"{n},{v},{x},{y},0" for n, v, x, y in rows) + "\n")
    monkeypatch.setenv("TN_REAL_ROOT", str(tmp_path))
    root = nb.find_real_root()
    assert root == tmp_path
    real = nb.load_tennis_dataset(root)
    assert len(real["frames"]) == 1 and real["frames"][0].shape == (6, nb.H, nb.W, 3)   # trimmed to a multiple of 3
    lab = real["labels"][0]
    np.testing.assert_allclose(lab[:, 2], [1, 0, 2, 1, 1, 1])
    np.testing.assert_allclose(lab[0, :2], [640 * nb.W / 1280, 360 * nb.H / 720], rtol=1e-5)
    np.testing.assert_array_equal(real["vis_raw"][0], [1, 0, 3, 2, 1, 1])     # TrackNet's original codes kept
    assert real["clips"] == ["game1/Clip1"]
    assert lab[1, 3] == 0 and lab[4, 3] > 0                                   # speed only between labelled frames


def _fake_tracknet_dataset(root, games=range(1, 11), clips=2, frames=9, size=(320, 180)):
    """Tiny TrackNet-format dataset: a bright ball moving right on a grey court, labelled at 1280x720 scale."""
    w0, h0 = size
    for g in games:
        for k in range(1, clips + 1):
            d = root / f"game{g}" / f"Clip{k}"
            d.mkdir(parents=True)
            rows = []
            for t in range(frames):
                img = np.full((h0, w0, 3), 90, np.uint8)
                x, y = 40 + 12 * t + 3 * g, 60 + 4 * k
                vis = 0 if t == frames - 1 else 1
                if vis:
                    cv2.circle(img, (x, y), 3, (60, 230, 230), -1)
                cv2.imwrite(str(d / f"{t:04d}.jpg"), img)
                rows.append(f"{t:04d}.jpg,{vis},{x if vis else ''},{y if vis else ''},0")
            (d / "Label.csv").write_text("file name,visibility,x-coordinate,y-coordinate,status\n" + "\n".join(rows) + "\n")


def test_real_game_level_split(nb, tmp_path):
    _fake_tracknet_dataset(tmp_path, clips=3)
    for k in (10, 2):                                     # numeric, not lexicographic, clip order
        (tmp_path / "game1" / f"Clip{k}").mkdir(exist_ok=True)
    real = nb.split_real(tmp_path)
    tr, va, te = (real[k]["clips"] for k in ("train", "val", "test"))
    assert {c.split("/")[0] for c in te} <= {"game9", "game10"}
    assert not ({c.split("/")[0] for c in tr + va} & {"game9", "game10"})   # no test match leaks into training
    assert not set(tr) & set(va)
    assert all(c.endswith("Clip3") for c in va)            # last clip of each training game is validation
    assert len(real["train"]["frames"]) == len(real["train"]["labels"]) == len(real["train"]["vis_raw"]) == len(tr)


def test_window_dataset_variable_length_clips(nb):
    rng = np.random.default_rng(0)
    lens = [3, 7, 12]
    split = dict(frames=[rng.integers(0, 255, (n, nb.H, nb.W, 3), dtype=np.uint8) for n in lens],
                 labels=[np.tile(np.array([[50, 40, 1, 0, 2, 0]], np.float32), (n, 1)) for n in lens])
    ds = nb.WindowDataset(split, "gauss")
    assert len(ds) == sum(n - 2 for n in lens)
    x, y, lab = ds[len(ds) - 1]
    np.testing.assert_allclose(x[6:9].numpy(), split["frames"][2][11].transpose(2, 0, 1) / 255.0, atol=1e-6)


def test_variant_tags_are_unique_and_safe(nb):
    tags = [nb.variant_tag(n) for n in nb.VARIANTS]
    assert len(set(tags)) == len(tags)
    assert all(ch.isalnum() or ch in "_-" for t in tags for ch in t)


def test_cost_table(nb):
    c = nb.COST
    assert (c["params_M"] > 0).all() and (c["GFLOPs"] > 0).all()
    v2, v5 = c.loc["V2 (baseline)"], c.loc["V5 (faithful)"]
    assert 1.0 < v5.GFLOPs / v2.GFLOPs < 1.15        # +3.3% at 512x288 (paper: +3.7%); relatively more at 288x160
    assert c.loc["V5-Lite+ (M1–M4)"].GFLOPs < 0.3 * v5.GFLOPs


# ----------------------------------------------------------------------------------------------
# slow: learnability + full notebook execution
# ----------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("name", ["V2 (baseline)", "V5-Lite+ (M1–M4)"])
def test_tiny_overfit(nb, name):
    """~20 s sanity check that gradients reach the ball: overfit 4 windows and localise the ball."""
    torch.manual_seed(0)
    spec = nb.VARIANTS[name]
    ds = nb.WindowDataset(nb.DATA["train"], spec["target"], augment=False)
    idx = [k for k in range(len(ds)) if ds[k][2][:, 2].min() == 1][:4]
    xb = torch.stack([ds[k][0] for k in idx]); yb = torch.stack([ds[k][1] for k in idx])
    lb = torch.stack([ds[k][2] for k in idx]).numpy()
    m = nb.TrackNet(**spec)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(120):
        m.train()
        loss = nb.wbce_loss(m(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.3 * np.mean(losses[:10])
    m.eval()
    with torch.no_grad():
        hm = torch.sigmoid(m(xb)).numpy()
    errs = []
    for b in range(len(idx)):
        for k in range(3):
            xy, _ = nb.DECODERS[spec["decode"]](hm[b, k], 0.3)
            errs.append(np.inf if xy is None else math.hypot(xy[0] - lb[b, k, 0], xy[1] - lb[b, k, 1]))
    assert np.mean(np.array(errs) <= nb.CFG["tol"]) >= 0.75, errs


@pytest.mark.slow
@pytest.mark.parametrize("profile,real", [("cpu", False), ("gpu", False), ("cpu", True)])
def test_notebook_runs_end_to_end(tmp_path, profile, real):
    """Executes every cell (smoke-sized data, 4 training steps per model) for both profiles, with and without
    a (fake, tiny) real TrackNet dataset attached, so the fine-tuning / real-evaluation path runs too."""
    import shutil
    nb_copy = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, nb_copy)
    env = dict(os.environ, TN_SMOKE="1", TN_PROFILE=profile, TN_ZIP="1")
    env.pop("TN_REAL_ROOT", None)
    if real:
        _fake_tracknet_dataset(tmp_path / "fake_tennis")
        env["TN_REAL_ROOT"] = str(tmp_path / "fake_tennis")
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        str(nb_copy), "--ExecutePreprocessor.timeout=1800"], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]
    figs = sorted(p.name for p in (tmp_path / "figures").glob("*"))
    assert "tracking_demo.gif" in figs and ("fig20_real_dataset.png" in figs) == real, figs
    check = json.loads((tmp_path / "results" / f"report_assets_check_{profile}.json").read_text())
    assert check["missing"] == [] and check["real_data"] == real, check     # every asset the report needs exists
    assert (tmp_path / "results" / "REPORT_ASSETS.md").exists()
    for f in ("ablation", "changes_summary", "changes_measured", "all_results") + (("real_tennis", "real_changes_summary") if real else ()):
        assert any((tmp_path / "results").glob(f"{f}_{profile}.*")), f
    if real:
        md = (tmp_path / "results" / f"changes_measured_{profile}.md").read_text(encoding="utf-8")
        assert "Real TrackNet tennis footage" in md
    assert (tmp_path / "tracknet_outputs.zip").stat().st_size > 0
