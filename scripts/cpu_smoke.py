"""Reproducible CPU smoke check: train (--fast-dev-run) -> evaluate -> predict, in one go.

    python scripts/cpu_smoke.py                          # synthetic images, ~10 s, no data needed
    python scripts/cpu_smoke.py --data-root path/to/PlantVillage --backbone resnet18

Without ``--data-root`` it writes a tiny synthetic ImageFolder (3 classes x 30
images) to a temp folder. It runs the same three command lines you would run by
hand, prints their output, and exits non-zero if any step fails. It checks the
plumbing, not the model: accuracy from a fast-dev-run means nothing. Outputs go
to ``--out-dir`` (default: a temp folder, printed at the end).

Needs torch, torchvision and matplotlib (CPU builds are fine).
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SYNTHETIC_CLASSES = ["Tomato_healthy", "Tomato_Late_blight", "Tomato_rust"]


def make_synthetic_imagefolder(root: Path, images_per_class: int = 30) -> Path:
    """One folder per class, each with a distinct colour so the task is learnable."""
    from PIL import Image

    for c, cls in enumerate(SYNTHETIC_CLASSES):
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new("RGB", (40, 30), color=(c * 100, 200 - c * 60, (i * 7) % 256))
            img.save(cls_dir / f"img_{i:02d}.jpg")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None, help="real ImageFolder root (default: synthetic images)")
    parser.add_argument("--backbone", default="baseline", choices=["baseline", "resnet18", "efficientnet_b0"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from leafguard import evaluate, predict, train

    work = Path(tempfile.mkdtemp(prefix="leafguard_smoke_"))
    data_root = Path(args.data_root) if args.data_root else make_synthetic_imagefolder(work / "data")
    out_dir = Path(args.out_dir) if args.out_dir else work / "outputs"
    weights = out_dir / f"{args.backbone}_best.pt"
    images = (p for p in sorted(data_root.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sample_image = next(images)

    steps = [
        ("train", train.main, [
            "--backbone", args.backbone, "--fast-dev-run", "--img-size", str(args.img_size),
            "--data-root", str(data_root), "--out-dir", str(out_dir),
        ]),
        ("evaluate", evaluate.main, [
            "--backbone", args.backbone, "--weights", str(weights), "--data-root", str(data_root),
            "--out-dir", str(out_dir), "--num-workers", "0",
        ]),
        ("predict", predict.main, [str(sample_image), "--weights", str(weights)]),
    ]
    for name, step, step_args in steps:
        print(f"\n=== {name}: python -m leafguard.{name} {' '.join(step_args)}")
        t0 = time.time()
        step(step_args)
        print(f"--- {name} ok ({time.time() - t0:.1f}s)")

    print(f"\nSmoke check passed. Outputs in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
