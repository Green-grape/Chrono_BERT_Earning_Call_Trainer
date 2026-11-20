from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import model_id
from .config import ROOT, load_config
from .data import prepare_year
from .export import export_checkpoint, resolve_export


def parser():
    ap = argparse.ArgumentParser(
        description="2013–2024 ChronoBERT: independent Presentation/Q&A DAPT"
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    sub = ap.add_subparsers(dest="command", required=True)
    # 하나의 프로그램을 여러개의 기능으로 나눠서 수행
    for name in ("plan", "check", "prepare", "train", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument(
            "--years", type=int, nargs="+", help="Subset of configured checkpoint years"
        )
        cmd.add_argument(
            "--sections", choices=["presentation", "qna"], nargs="+"
        )  # 하나 이상의 리스트를 받아서 저장 (--sections qna presentation)
        if name in ("train", "run"):
            cmd.add_argument(
                "--resume",
                action="store_true",
                help="Resume latest matching job; completed jobs re-export",
            )
            cmd.add_argument(
                "--smoke-steps",
                type=int,
                default=0,
                help="Stop each job after N optimizer steps",
            )
            cmd.add_argument(
                "--target-tokens",
                type=int,
                help="Cumulative token goal (at least the configured goal); extends --resume",
            )
            cmd.add_argument(
                "--allow-download",
                action="store_true",
                help="Allow base model downloads (default offline)",
            )
    cmd = sub.add_parser(
        "export", help="Export an existing checkpoint without loading model weights"
    )
    cmd.add_argument("--checkpoint", required=True, type=Path)
    cmd.add_argument("--year", required=True, type=int)
    cmd.add_argument("--section", required=True, choices=["presentation", "qna"])
    cmd = sub.add_parser(
        "resolve", help="Verify and print a portable exported snapshot path"
    )
    cmd.add_argument("--year", required=True, type=int)
    cmd.add_argument("--section", required=True, choices=["presentation", "qna"])
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "allow_download", False):
        cfg.local_files_only = False
    if args.command == "resolve":
        print(
            resolve_export(Path(cfg.output_root) / "exports", args.year, args.section)
        )
        return
    if args.command == "export":
        print(
            export_checkpoint(
                args.checkpoint.resolve(),
                Path(cfg.output_root) / "exports",
                args.year,
                args.section,
            )
        )
        return
    years = args.years or list(range(cfg.start_year, cfg.end_year + 1))
    if len(set(years)) != len(years) or any(
        not cfg.start_year <= y <= cfg.end_year for y in years
    ):
        ap.error("--years must be unique and within configured start_year/end_year")
    if args.sections:
        cfg.sections = tuple(args.sections)
    cfg.validate()
    if getattr(args, "smoke_steps", 0) < 0:
        ap.error("--smoke-steps must be nonnegative")
    if getattr(args, "target_tokens", None) is not None and args.target_tokens < cfg.target_tokens:
        ap.error("--target-tokens must be at least the configured target_tokens")
    if args.command == "check":
        from .check import preflight

        report = preflight(cfg, years)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["all_models_cached"]:
            raise SystemExit(1)
        return
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "config": cfg.to_dict(),
                    "jobs": [
                        {
                            "year": year,
                            "base_model": model_id(year),
                            "section": section,
                            "cutoff": f"{year}-12-31",
                            "target_tokens": cfg.target_tokens,
                        }
                        for year in years
                        for section in cfg.sections
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for year in years:
        # train and run both validate/prepare reusable data, so invocation order is safe.
        data_dir = prepare_year(cfg, year)
        print(f"Prepared {year}: {data_dir}", flush=True)
        if args.command == "prepare":
            continue
        from .training import train_job

        for section in cfg.sections:
            result = train_job(
                cfg,
                year,
                section,
                data_dir,
                resume=args.resume,
                smoke_steps=args.smoke_steps,
                target_tokens=args.target_tokens,
            )
            print(f"Result {year}/{section}: {result}", flush=True)
