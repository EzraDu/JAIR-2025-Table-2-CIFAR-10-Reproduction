"""Replay standard CMI ZSKT (--epochs 200) through epoch 101 and save that epoch.

Stops before epoch 102. Does not edit CMI. Restores the student checkpoint on exit.
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

CMI = Path(os.environ.get("CMI_ROOT", "CMI"))
OUT = CMI / "run" / "zskt_stages" / "epoch101_calls"
REFERENCE = CMI / "run" / "zskt_stages" / "std" / "epoch_101.npy"
CKPT = CMI / "checkpoints" / "datafree-zskt" / "cifar10-wrn40_2-wrn16_1.pth"
BACKUP = Path(os.environ.get("ZSKT_STUDENT_BACKUP", "backup_zskt_student_wrn16_1.pth"))
CKPT_BYTES = 1448501
TARGET_EPOCH = 101
EPOCH0_MEAN = 0.584603
# Printed LR during epoch 101 of the measured 200-epoch run was 0.0492.
EPOCH101_LR_LO = 0.045
EPOCH101_LR_HI = 0.055
CMI_LOG = CMI / "checkpoints" / "datafree-zskt" / "log-cifar10-wrn40_2-wrn16_1-e101t200.txt"


class StopAfterTarget(Exception):
    pass

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
BASE = [
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
]


def protect_checkpoint():
    if not BACKUP.exists():
        size = CKPT.stat().st_size
        if size != CKPT_BYTES:
            raise SystemExit(f"refusing to backup {CKPT} size {size}, expected {CKPT_BYTES}")
        shutil.copy2(CKPT, BACKUP)
        print(f"BACKED_UP {BACKUP}", flush=True)
    elif BACKUP.stat().st_size != CKPT_BYTES:
        raise SystemExit(f"backup size {BACKUP.stat().st_size} != {CKPT_BYTES}")


def restore_checkpoint():
    if BACKUP.exists() and BACKUP.stat().st_size == CKPT_BYTES:
        shutil.copy2(BACKUP, CKPT)
        print(f"RESTORED {CKPT}", flush=True)


def save_uint8(images, dest):
    pix = images.detach()
    clamped = pix.clamp(0, 1)
    pix_mean = float(clamped.mean())
    pix_std = float(clamped.std())
    arr = clamped.mul(255).round().to(torch.uint8).cpu().numpy()
    np.save(dest, arr)
    return arr.shape[0], pix_mean, pix_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if OUT.exists():
        if not args.replace:
            raise SystemExit(f"refusing to mix with existing {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "run.json").write_text(json.dumps({
        "source": "CMI GenerativeSynthesizer zskt, standard 200-epoch cosine schedule",
        "target_epoch": TARGET_EPOCH,
        "epochs_arg": 200,
        "stopped_after_epoch": TARGET_EPOCH,
        "adv": 1, "bn": 0, "oh": 0, "batch_size": 256,
        "note": "Every synthesize call of epoch 101 is saved, then training stops before epoch 102. Published zskt images are not touched.",
    }, indent=2), encoding="utf-8")

    calls_per_epoch = 400 // 5
    protect_checkpoint()
    counter = {"call": 0, "saved": 0}
    try:
        os.chdir(CMI)
        sys.path.insert(0, str(CMI))
        from datafree.synthesis.generative import GenerativeSynthesizer

        original = GenerativeSynthesizer.synthesize

        def wrapped(self):
            if counter["call"] >= (TARGET_EPOCH + 1) * calls_per_epoch:
                raise StopAfterTarget()
            result = original(self)
            call = counter["call"]
            counter["call"] = call + 1
            epoch = call // calls_per_epoch
            step = call % calls_per_epoch
            if epoch == 0 and step == calls_per_epoch - 1:
                _, pix_mean, pix_std = save_uint8(result["synthetic"], OUT / "epoch0_last.npy")
                print(f"EPOCH0_CHECK pix_mean={pix_mean:.6f} pix_std={pix_std:.6f}", flush=True)
                if abs(pix_mean - EPOCH0_MEAN) > 0.01:
                    print(
                        f"DIVERGED at epoch 0: pix_mean={pix_mean:.6f}, expected {EPOCH0_MEAN:.6f}",
                        flush=True,
                    )
                    raise SystemExit(5)
            if epoch != TARGET_EPOCH:
                return result
            dest = OUT / f"step_{step:02d}.npy"
            n, pix_mean, pix_std = save_uint8(result["synthetic"], dest)
            counter["saved"] += 1
            print(
                f"SNAP epoch={epoch} step={step} call={call} n={n} "
                f"pix_mean={pix_mean:.4f} pix_std={pix_std:.4f}",
                flush=True,
            )
            return result

        GenerativeSynthesizer.synthesize = wrapped
        sys.argv = [
            "datafree_kd.py",
            *COMMON,
            *BASE,
            "--save_dir", str(OUT / "synthesis"),
            "--log_tag", "e101t200",
        ]
        print("COLLECT_START", " ".join(sys.argv), flush=True)
        import datafree_kd
        try:
            datafree_kd.main()
        except StopAfterTarget:
            print(f"STOP_AFTER_EPOCH {TARGET_EPOCH}", flush=True)
        else:
            print(
                f"UNEXPECTED_CALLS calls={counter['call']} saved={counter['saved']} ran_to_end=True",
                flush=True,
            )
            raise SystemExit(4)
        if counter["saved"] != calls_per_epoch:
            print(
                f"UNEXPECTED_CALLS calls={counter['call']} saved={counter['saved']}",
                flush=True,
            )
            raise SystemExit(4)
        for logger in logging.root.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                for handler in logger.handlers:
                    handler.flush()
        if not CMI_LOG.exists():
            print(f"BAD_LR missing log {CMI_LOG}", flush=True)
            raise SystemExit(6)
        lr_hits = re.findall(
            r"\[Eval\] Epoch=101 .*?Lr=([0-9.]+)",
            CMI_LOG.read_text(encoding="utf-8", errors="replace"),
        )
        if not lr_hits:
            print("BAD_LR missing Epoch=101 line", flush=True)
            raise SystemExit(6)
        epoch_lr = float(lr_hits[-1])
        print(f"EPOCH101_LR {epoch_lr:.4f}", flush=True)
        if not (EPOCH101_LR_LO <= epoch_lr <= EPOCH101_LR_HI):
            print(
                f"BAD_LR {epoch_lr:.4f} expected about 0.0492 for T_max=200",
                flush=True,
            )
            raise SystemExit(6)
        last = np.load(OUT / "step_79.npy")
        ref = np.load(REFERENCE)
        diff = int(np.max(np.abs(last.astype(np.int16) - ref.astype(np.int16))))
        print(f"EPOCH101_MATCH max_abs_uint8={diff} equal={diff == 0}", flush=True)
        print(f"COLLECT_DONE saved={counter['saved']} epoch={TARGET_EPOCH}", flush=True)
    finally:
        restore_checkpoint()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        os._exit(code)
    os._exit(0)
