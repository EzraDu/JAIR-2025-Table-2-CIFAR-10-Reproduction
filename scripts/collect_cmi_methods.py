"""Collect 10240 inverted CIFAR-10 images per CMI-repo method.

The CMI repository is not modified. This script only passes the flags already
defined in datafree_kd.py. Generative methods (dafl, dfq, zskt) never write an
image pool; their official synthesize() return value is saved from this process.
DeepInversion already writes run/<method>/*.png. Its internal DataLoader is
forced to 0 workers from this process so Windows does not deadlock.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

CMI_ROOT = Path(os.environ.get("CMI_ROOT", "CMI"))
INV_ROOT = Path(os.environ.get("INVDET_ROOT", "invdet")) / "Invdata" / "cifar10"
LIMIT = 10240

# Arguments taken from scripts/<method>/*_wrn402_wrn161.sh.
# Those shell files are truncated after --seed, so save_dir / workers are set
# here via flags the Python entry point already accepts.
COMMON = [
    "--dataset", "cifar10",
    "--teacher", "wrn40_2",
    "--student", "wrn16_1",
    "--lr", "0.1",
    "--gpu", "0",
    "--seed", "0",
    "--T", "20",
    "-j", "0",
]

METHOD_ARGS = {
    "dafl": [
        "--method", "dafl",
        "--batch_size", "256",
        "--epochs", "200",
        "--kd_steps", "5",
        "--ep_steps", "400",
        "--g_steps", "1",
        "--lr_g", "1e-3",
        "--adv", "0",
        "--bn", "0",
        "--oh", "1",
        "--act", "0.001",
        "--balance", "20",
    ],
    "dfq": [
        "--method", "dfq",
        "--batch_size", "256",
        "--epochs", "200",
        "--kd_steps", "10",
        "--ep_steps", "400",
        "--g_steps", "1",
        "--lr_g", "1e-3",
        "--adv", "1",
        "--bn", "1",
        "--oh", "1",
        "--act", "0",
        "--balance", "20",
    ],
    "zskt": [
        "--method", "zskt",
        "--batch_size", "256",
        "--epochs", "200",
        "--kd_steps", "5",
        "--ep_steps", "400",
        "--g_steps", "1",
        "--lr_g", "1e-3",
        "--adv", "1",
        "--bn", "0",
        "--oh", "0",
        "--act", "0",
        "--balance", "0",
    ],
    "deepinv": [
        "--method", "deepinv",
        "--batch_size", "256",
        "--epochs", "40",
        "--kd_steps", "400",
        "--ep_steps", "400",
        "--g_steps", "1000",
        "--lr_g", "0.1",
        "--adv", "1",
        "--bn", "10",
        "--oh", "1",
        "--act", "0",
        "--balance", "0",
    ],
    # CMISynthesizer on wrn40_2 (scratch CMI flags).
    # 40 epochs x synthesis batch 256 = 10240 images.
    "ipr": [
        "--method", "cmi",
        "--batch_size", "128",
        "--synthesis_batch_size", "256",
        "--epochs", "40",
        "--kd_steps", "400",
        "--ep_steps", "400",
        "--g_steps", "200",
        "--lr_g", "1e-3",
        "--adv", "0.5",
        "--bn", "1.0",
        "--oh", "1.0",
        "--cr", "0.8",
        "--cr_T", "0.1",
        "--act", "0",
        "--balance", "0",
    ],
}


def with_epochs(args, epochs):
    out = []
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token == "--epochs":
            skip = True
            continue
        out.append(token)
    out.extend(["--epochs", str(epochs)])
    return out


def publish(src: Path, method: str):
    files = sorted(p for p in src.rglob("*.png") if p.is_file())
    if not files:
        raise SystemExit(f"no png written for {method} under {src}")
    if len(files) > LIMIT:
        # Keep a deterministic spread across the whole run, not only the prefix.
        import random
        rng = random.Random(0)
        files = sorted(rng.sample(files, LIMIT))
    dst = INV_ROOT / method / "0"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for i, src_path in enumerate(files):
        shutil.copyfile(src_path, dst / f"{i:05d}.png")
    print(f"PUBLISHED {method}: {len(list(dst.glob('*.png')))} -> {dst}", flush=True)


def run_pooled(method, epochs):
    # DeepInvSyntheiszer and CMISynthesizer build their own DataLoader with
    # num_workers=4. On Windows that worker pool deadlocks after the first
    # synthesized batch. Force workers to 0; the CMI sources stay unchanged.
    os.chdir(CMI_ROOT)
    if str(CMI_ROOT) not in sys.path:
        sys.path.insert(0, str(CMI_ROOT))
    import torch.utils.data as data_utils

    original_loader = data_utils.DataLoader

    def loader_no_workers(*args, **kwargs):
        kwargs["num_workers"] = 0
        return original_loader(*args, **kwargs)

    data_utils.DataLoader = loader_no_workers
    save_dir = CMI_ROOT / "run" / method
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)
    sys.argv = [
        "datafree_kd.py",
        *COMMON,
        *with_epochs(METHOD_ARGS[method], epochs),
        "--save_dir", str(save_dir),
        "--log_tag", method,
    ]
    print("RUN", " ".join(sys.argv), flush=True)
    import datafree_kd
    datafree_kd.main()
    if not getattr(main, "no_publish", False):
        publish(save_dir, method)


def run_generative(method, epochs):
    os.chdir(CMI_ROOT)
    if str(CMI_ROOT) not in sys.path:
        sys.path.insert(0, str(CMI_ROOT))
    from datafree.synthesis.generative import GenerativeSynthesizer

    tokens = with_epochs(METHOD_ARGS[method], epochs)
    def flag(name, default):
        if name not in tokens:
            return default
        return int(tokens[tokens.index(name) + 1])

    batch = flag("--batch_size", 256)
    calls_per_epoch = flag("--ep_steps", 400) // flag("--kd_steps", 1)
    need_calls = (LIMIT + batch - 1) // batch
    total_calls = epochs * calls_per_epoch
    start_at = max(0, total_calls - need_calls)
    print(
        f"[{method}] synthesize calls={total_calls}, "
        f"saving the last {need_calls} (from call {start_at})",
        flush=True,
    )

    scratch = CMI_ROOT / "run" / method / "_latest"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    counter = {"call": 0, "saved": 0}

    original = GenerativeSynthesizer.synthesize

    def wrapped(self):
        result = original(self)
        call = counter["call"]
        counter["call"] = call + 1
        if call < start_at:
            return result
        images = result["synthetic"]
        array = (images.detach().clamp(0, 1).mul(255).byte().cpu().numpy())
        from PIL import Image
        for image in array:
            Image.fromarray(image.transpose(1, 2, 0)).save(scratch / f"{counter['saved']:06d}.png")
            counter["saved"] += 1
            if counter["saved"] >= LIMIT:
                break
        print(f"[{method}] call {call + 1}/{total_calls} saved={counter['saved']}", flush=True)
        return result

    GenerativeSynthesizer.synthesize = wrapped
    sys.argv = [
        "datafree_kd.py",
        *COMMON,
        *with_epochs(METHOD_ARGS[method], epochs),
        "--save_dir", str(CMI_ROOT / "run" / method),
        "--log_tag", method,
    ]
    import datafree_kd
    datafree_kd.main()
    if not getattr(main, "no_publish", False):
        publish(scratch, method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHOD_ARGS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    main.no_publish = args.no_publish
    epochs = args.epochs
    if epochs is None:
        epochs = 40 if args.method in ("deepinv", "ipr") else 200
    if args.method in ("deepinv", "ipr"):
        run_pooled(args.method, epochs)
    else:
        run_generative(args.method, epochs)


if __name__ == "__main__":
    main()
