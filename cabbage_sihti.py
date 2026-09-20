#!/usr/bin/env python3
"""CabbageFarmSihti v0: learn a stationary spectral law, sample it continuously.

This is deliberately the strongest simple attacker before adding nonlinear
Sihti cross-octave structure. It learns a 2-D spectral distribution in a PCA
colour basis, then realizes a globally coordinate-addressable random Fourier
field from that distribution.

Examples:
    python cabbage_sihti.py fit texture.jpg model.json
    python cabbage_sihti.py synth model.json out.png --width 768 --height 512
    python cabbage_sihti.py compare texture.jpg out.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_rgb(path: str | Path, max_size: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_size and max(im.size) > max_size:
        scale = float(max_size) / max(im.size)
        im = im.resize(
            (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(im, dtype=np.float64) / 255.0


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    x = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    Image.fromarray(np.round(255.0 * x).astype(np.uint8), "RGB").save(path)


def fit_colour_basis(rgb: np.ndarray):
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    mean = flat.mean(axis=0)
    centered = flat - mean
    cov = centered.T @ centered / max(len(flat) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    basis = vectors[:, order]
    # Fix arbitrary eigenvector signs for stable saved models.
    for k in range(3):
        j = int(np.argmax(np.abs(basis[:, k])))
        if basis[j, k] < 0:
            basis[:, k] *= -1
    coeff = centered @ basis
    return mean, basis, coeff.reshape(rgb.shape)


def hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    win = wy[:, None] * wx[None, :]
    norm = np.sqrt(np.mean(win * win))
    return win / max(norm, 1e-12)


def spectral_measure(channel: np.ndarray, max_bins: int = 4096):
    """Return a sparse discrete approximation to a 2-D spectral measure.

    Bins are full FFT frequencies. A cosine realization implicitly supplies the
    conjugate partner, so it is okay that the learned distribution is symmetric.
    """
    x = np.asarray(channel, dtype=np.float64)
    h, w = x.shape
    x = x - x.mean()
    X = np.fft.fft2(x * hann2d(h, w))
    power = np.abs(X) ** 2
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")

    mask = np.ones((h, w), dtype=bool)
    mask[0, 0] = False
    p = power[mask].ravel()
    xs = FX[mask].ravel()
    ys = FY[mask].ravel()

    if max_bins and p.size > max_bins:
        # Keep high-energy bins, but reserve some support for the tail so the
        # model does not become a deterministic "top frequencies only" codec.
        n_top = int(max_bins * 0.75)
        n_tail = max_bins - n_top
        order = np.argpartition(p, -n_top)[-n_top:]
        rest_mask = np.ones(p.size, dtype=bool)
        rest_mask[order] = False
        rest = np.flatnonzero(rest_mask)
        if rest.size and n_tail:
            # Deterministic quantile coverage of the remaining cumulative power.
            rp = p[rest]
            ro = rest[np.argsort(rp)]
            c = np.cumsum(p[ro])
            if c[-1] > 0:
                targets = np.linspace(0, c[-1], n_tail + 2)[1:-1]
                pick = ro[np.searchsorted(c, targets, side="left")]
                keep = np.unique(np.concatenate([order, pick]))
            else:
                keep = order
        else:
            keep = order
        p, xs, ys = p[keep], xs[keep], ys[keep]

    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if total <= 0:
        # Constant channel: keep one harmless near-zero frequency.
        return [{"fx": 0.0, "fy": 0.0, "p": 1.0}]
    p /= total
    return [
        {"fx": float(a), "fy": float(b), "p": float(q)}
        for a, b, q in zip(xs, ys, p)
        if q > 0
    ]


def gaussian_residue_signature(coeff: np.ndarray, times=(0.5, 1, 2, 4, 8, 16)):
    """Linear scale-space signature.

    This is intentionally a control. For a stationary Gaussian process these
    second-order octave statistics are determined by the power spectrum.
    """
    x = np.asarray(coeff, dtype=np.float64)
    records = []
    residues = []
    for t in times:
        s1 = math.sqrt(2.0 * float(t))
        s2 = math.sqrt(4.0 * float(t))
        u1 = gaussian_filter(x, sigma=(s1, s1, 0), mode="reflect")
        u2 = gaussian_filter(x, sigma=(s2, s2, 0), mode="reflect")
        r = u1 - u2
        residues.append(r)
        rms = np.sqrt(np.mean(r * r, axis=(0, 1)))
        records.append({"time": float(t), "rms": [float(v) for v in rms]})

    for j, rec in enumerate(records):
        rec["corr_next"] = None
        if j + 1 < len(residues):
            a = residues[j].reshape(-1, 3)
            b = residues[j + 1].reshape(-1, 3)
            corrs = []
            for c in range(3):
                aa = a[:, c] - a[:, c].mean()
                bb = b[:, c] - b[:, c].mean()
                den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
                corrs.append(float(aa @ bb / den) if den > 1e-12 else 0.0)
            rec["corr_next"] = corrs
    return records


def fit_model(rgb: np.ndarray, max_bins: int = 4096):
    rgb = np.asarray(rgb, dtype=np.float64)
    h, w = rgb.shape[:2]
    mean, basis, coeff = fit_colour_basis(rgb)
    comps = []
    for c in range(3):
        variance = float(np.var(coeff[..., c]))
        comps.append(
            {
                "variance": variance,
                "spectral_measure": spectral_measure(coeff[..., c], max_bins=max_bins),
            }
        )
    return {
        "format": "cabbage-farm-sihti-v0",
        "reference_shape": [int(h), int(w)],
        "colour_mean": [float(v) for v in mean],
        "colour_basis": [[float(v) for v in row] for row in basis],
        "components": comps,
        "gaussian_residue_signature": gaussian_residue_signature(coeff),
        "notes": {
            "generator": "stationary Gaussian random Fourier field",
            "boundary": "power spectrum / second-order baseline; not a higher-order texture model",
        },
    }


def _component_modes(component: dict, ref_shape, mode_count: int, rng):
    bins = component["spectral_measure"]
    p = np.array([b["p"] for b in bins], dtype=np.float64)
    p /= p.sum()
    idx = rng.choice(len(bins), size=int(mode_count), replace=True, p=p)
    h, w = ref_shape
    # Jitter inside one DFT cell to remove exact reference-period tiling.
    fx = np.array([bins[i]["fx"] for i in idx]) + rng.uniform(-0.5 / w, 0.5 / w, len(idx))
    fy = np.array([bins[i]["fy"] for i in idx]) + rng.uniform(-0.5 / h, 0.5 / h, len(idx))
    phase = rng.uniform(0.0, 2.0 * np.pi, len(idx))
    variance = max(float(component["variance"]), 0.0)
    amp = math.sqrt(2.0 * variance / max(len(idx), 1))
    return fx, fy, phase, amp


def synthesize(
    model: dict,
    width: int,
    height: int,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    scale: float = 1.0,
    seed: int = 0,
    modes: int = 512,
    chunk_modes: int = 64,
):
    """Evaluate one globally defined random field on an arbitrary crop."""
    width, height = int(width), int(height)
    yy, xx = np.meshgrid(
        origin_y + scale * np.arange(height, dtype=np.float64),
        origin_x + scale * np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    flat_x, flat_y = xx.ravel(), yy.ravel()
    coeff = np.zeros((flat_x.size, 3), dtype=np.float64)
    ref_shape = tuple(model["reference_shape"])
    base_rng = np.random.SeedSequence(int(seed))
    child_seeds = base_rng.spawn(3)

    for c in range(3):
        rng = np.random.default_rng(child_seeds[c])
        fx, fy, phase, amp = _component_modes(model["components"][c], ref_shape, modes, rng)
        acc = np.zeros(flat_x.size, dtype=np.float64)
        for start in range(0, len(fx), int(chunk_modes)):
            sl = slice(start, min(start + int(chunk_modes), len(fx)))
            arg = (
                2.0
                * np.pi
                * (
                    fx[sl, None] * flat_x[None, :]
                    + fy[sl, None] * flat_y[None, :]
                )
                + phase[sl, None]
            )
            acc += np.cos(arg).sum(axis=0)
        coeff[:, c] = amp * acc

    basis = np.asarray(model["colour_basis"], dtype=np.float64)
    mean = np.asarray(model["colour_mean"], dtype=np.float64)
    rgb = coeff @ basis.T + mean
    return np.clip(rgb.reshape(height, width, 3), 0.0, 1.0)


def radial_psd(rgb: np.ndarray, bins: int = 48):
    x = np.asarray(rgb, dtype=np.float64)
    gray = x.mean(axis=2)
    gray -= gray.mean()
    h, w = gray.shape
    X = np.fft.fft2(gray * hann2d(h, w))
    P = np.abs(X) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    edges = np.linspace(0, r.max() + 1e-12, int(bins) + 1)
    out = np.zeros(int(bins), dtype=np.float64)
    count = np.zeros(int(bins), dtype=np.float64)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, int(bins) - 1)
    np.add.at(out, idx, P.ravel())
    np.add.at(count, idx, 1)
    out /= np.maximum(count, 1)
    out /= max(out.sum(), 1e-15)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, out


def gradient_anisotropy(rgb: np.ndarray):
    gray = np.asarray(rgb, dtype=np.float64).mean(axis=2)
    gy, gx = np.gradient(gray)
    C = np.array(
        [
            [np.mean(gx * gx), np.mean(gx * gy)],
            [np.mean(gx * gy), np.mean(gy * gy)],
        ]
    )
    vals = np.linalg.eigvalsh(C)
    return float(vals[-1] / max(vals[0], 1e-15))


def compare_images(a: np.ndarray, b: np.ndarray):
    # Crop to shared size for direct statistic comparison.
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]

    _, pa = radial_psd(a)
    _, pb = radial_psd(b)
    eps = 1e-12
    log_psd_rmse = float(np.sqrt(np.mean((np.log(pa + eps) - np.log(pb + eps)) ** 2)))

    ma, ba, ca = fit_colour_basis(a)
    mb, bb, cb = fit_colour_basis(b)
    # Compare RGB covariance, not PCA axes (which may swap under near-degeneracy).
    fa, fb = a.reshape(-1, 3), b.reshape(-1, 3)
    cova = np.cov(fa, rowvar=False)
    covb = np.cov(fb, rowvar=False)
    cov_rel = float(np.linalg.norm(cova - covb) / max(np.linalg.norm(cova), 1e-12))

    siga = gaussian_residue_signature(ca)
    sigb = gaussian_residue_signature((fb - mb) @ bb if False else cb)
    # flatten normalized residue energies
    ea = np.array([r["rms"] for r in siga], dtype=np.float64)
    eb = np.array([r["rms"] for r in sigb], dtype=np.float64)
    ea /= max(np.linalg.norm(ea), 1e-12)
    eb /= max(np.linalg.norm(eb), 1e-12)
    residue_energy_rmse = float(np.sqrt(np.mean((ea - eb) ** 2)))

    return {
        "log_radial_psd_rmse": log_psd_rmse,
        "gradient_anisotropy_a": gradient_anisotropy(a),
        "gradient_anisotropy_b": gradient_anisotropy(b),
        "rgb_covariance_relative_error": cov_rel,
        "gaussian_residue_energy_rmse": residue_energy_rmse,
        "reference_residue_signature": siga,
        "sample_residue_signature": sigb,
    }


def save_model(path: str | Path, model: dict):
    Path(path).write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_model(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def synthetic_reference(size=128, seed=0):
    """A deterministic nontrivial test texture: oriented multi-scale Gaussian field."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((size, size))
    broad = gaussian_filter(n, sigma=(3.0, 12.0), mode="wrap")
    fine = gaussian_filter(rng.standard_normal((size, size)), sigma=(0.8, 3.0), mode="wrap")
    z = 0.75 * broad / max(broad.std(), 1e-12) + 0.25 * fine / max(fine.std(), 1e-12)
    z = (z - z.mean()) / max(z.std(), 1e-12)
    rgb = np.stack(
        [
            0.50 + 0.14 * z,
            0.46 + 0.10 * z + 0.02 * n,
            0.40 + 0.07 * z,
        ],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def cmd_fit(a):
    rgb = load_rgb(a.reference, max_size=a.max_size)
    model = fit_model(rgb, max_bins=a.max_bins)
    save_model(a.model, model)
    print(f"wrote {a.model}")


def cmd_synth(a):
    model = load_model(a.model)
    rgb = synthesize(
        model,
        a.width,
        a.height,
        origin_x=a.origin_x,
        origin_y=a.origin_y,
        scale=a.scale,
        seed=a.seed,
        modes=a.modes,
    )
    save_rgb(a.output, rgb)
    print(f"wrote {a.output}")


def cmd_compare(a):
    x = load_rgb(a.reference, max_size=a.max_size)
    y = load_rgb(a.sample, max_size=a.max_size)
    result = compare_images(x, y)
    print(json.dumps(result, indent=2))


def cmd_demo(a):
    ref = synthetic_reference(a.size, a.seed)
    model = fit_model(ref, max_bins=a.max_bins)
    out = synthesize(model, a.size * 2, a.size, seed=a.seed + 1, modes=a.modes)
    save_rgb(a.reference_out, ref)
    save_rgb(a.sample_out, out)
    save_model(a.model_out, model)
    print(f"wrote {a.reference_out}, {a.sample_out}, {a.model_out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit")
    f.add_argument("reference")
    f.add_argument("model")
    f.add_argument("--max-size", type=int, default=256)
    f.add_argument("--max-bins", type=int, default=4096)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("synth")
    s.add_argument("model")
    s.add_argument("output")
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--height", type=int, default=512)
    s.add_argument("--origin-x", type=float, default=0.0)
    s.add_argument("--origin-y", type=float, default=0.0)
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--modes", type=int, default=512)
    s.set_defaults(func=cmd_synth)

    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("sample")
    c.add_argument("--max-size", type=int, default=256)
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("demo")
    d.add_argument("--size", type=int, default=128)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--modes", type=int, default=512)
    d.add_argument("--max-bins", type=int, default=4096)
    d.add_argument("--reference-out", default="demo_reference.png")
    d.add_argument("--sample-out", default="demo_sample.png")
    d.add_argument("--model-out", default="demo_model.json")
    d.set_defaults(func=cmd_demo)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
