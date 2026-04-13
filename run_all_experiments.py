from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _build_cmd(
    script_name: str,
    plan_path: str,
    batch: int | None,
    imgsz: int | None,
    workers: int | None,
    device: str | None,
    resume: bool,
    skip_completed: bool,
) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve().parent / script_name), "--plan", plan_path]
    if batch is not None:
        cmd += ["--batch", str(batch)]
    if imgsz is not None:
        cmd += ["--imgsz", str(imgsz)]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if device is not None:
        cmd += ["--device", str(device)]
    if resume:
        cmd += ["--resume"]
    if skip_completed:
        cmd += ["--skip-completed"]
    return cmd


def _run_with_recovery(cmd: list[str], label: str, max_restarts: int) -> None:
    attempts = 0
    while True:
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            attempts += 1
            if attempts > max_restarts:
                raise
            print(f"[RECOVER] {label} failed (exit={e.returncode}). Retrying {attempts}/{max_restarts} in 5s...")
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation and baseline experiments sequentially or in parallel.")
    parser.add_argument(
        "--ablation-plan",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\experiments\ablation_plan.yaml",
    )
    parser.add_argument(
        "--baseline-plan",
        type=str,
        default=r"C:\Users\Administrator\Desktop\project\3\experiments\baseline_plan.yaml",
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch for both suites.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override imgsz for both suites.")
    parser.add_argument("--workers", type=int, default=None, help="Override workers for both suites.")
    parser.add_argument("--ablation-device", type=str, default=None, help="Device for ablation suite.")
    parser.add_argument("--baseline-device", type=str, default=None, help="Device for baseline suite.")
    parser.add_argument("--resume", action="store_true", help="Resume suites from checkpoints when possible.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip already-completed experiments.")
    parser.add_argument("--auto-recover", action="store_true", help="Auto-retry failed suite runs.")
    parser.add_argument("--max-restarts", type=int, default=3, help="Max retries when --auto-recover is enabled.")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run both suites simultaneously. Use different devices if possible.",
    )
    args = parser.parse_args()

    ablation_cmd = _build_cmd(
        script_name="run_ablation_suite.py",
        plan_path=args.ablation_plan,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.ablation_device,
        resume=args.resume or args.auto_recover,
        skip_completed=args.skip_completed or args.auto_recover,
    )
    baseline_cmd = _build_cmd(
        script_name="run_baseline_suite.py",
        plan_path=args.baseline_plan,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.baseline_device,
        resume=args.resume or args.auto_recover,
        skip_completed=args.skip_completed or args.auto_recover,
    )

    if args.parallel:
        log_dir = Path(__file__).resolve().parent / "runs_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ablation_log = (log_dir / "ablation_parallel.log").open("w", encoding="utf-8")
        baseline_log = (log_dir / "baseline_parallel.log").open("w", encoding="utf-8")
        p1 = subprocess.Popen(ablation_cmd, stdout=ablation_log, stderr=subprocess.STDOUT)
        p2 = subprocess.Popen(baseline_cmd, stdout=baseline_log, stderr=subprocess.STDOUT)
        code1 = p1.wait()
        code2 = p2.wait()
        ablation_log.close()
        baseline_log.close()
        if code1 != 0 or code2 != 0:
            raise SystemExit(f"Parallel run failed: ablation={code1}, baseline={code2}")
        print("Parallel run completed successfully.")
        print(f"Logs: {log_dir}")
        return

    if args.auto_recover:
        _run_with_recovery(ablation_cmd, "ablation", args.max_restarts)
        _run_with_recovery(baseline_cmd, "baseline", args.max_restarts)
    else:
        subprocess.run(ablation_cmd, check=True)
        subprocess.run(baseline_cmd, check=True)
    print("Sequential run completed successfully.")


if __name__ == "__main__":
    main()
