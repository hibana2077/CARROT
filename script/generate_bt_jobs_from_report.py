#!/usr/bin/env python3
"""Generate BT PBS job scripts from script/BT/report.csv.

Reads a CSV like:
  backbone,batch_size,img_size,seed,accuracy
and generates BT###.sh scripts by copying a template (default: script/BT/BT000.sh)
while replacing only:
  --model, --batch_size, --img_size (and --seed if present in CSV)

It also updates the log redirection (>> ... 2>&1) to use BT###.log.

Example:
  python script/generate_bt_jobs_from_report.py \
    --report script/BT/report.csv \
    --template script/BT/BT000.sh \
    --out-dir script/BT
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_LOG_RE = re.compile(r"(>>\s*)([^\s]+\.log)(\s+2>&1)")


def _flag_pattern(flag: str) -> re.Pattern[str]:
    # Matches: --flag <value>
    return re.compile(rf"(--{re.escape(flag)}\s+)([^\s\\]+)(\b)")


def _replace_flag_once(script_text: str, flag: str, new_value: str) -> str:
    pattern = _flag_pattern(flag)
    match = pattern.search(script_text)
    if not match:
        raise ValueError(f"Cannot find --{flag} in template.")
    return pattern.sub(lambda m: f"{m.group(1)}{new_value}{m.group(3)}", script_text, count=1)


def _replace_log(script_text: str, new_log_name: str) -> str:
    match = _LOG_RE.search(script_text)
    if not match:
        raise ValueError("Cannot find log redirection pattern like '>> something.log 2>&1' in template.")
    return _LOG_RE.sub(lambda m: f"{m.group(1)}{new_log_name}{m.group(3)}", script_text, count=1)


def _required(row: dict[str, str], key: str) -> str:
    val = (row.get(key) or "").strip()
    if not val:
        raise ValueError(f"Missing required column '{key}' in row: {row}")
    return val


def generate_from_report(
    template_path: Path,
    report_path: Path,
    out_dir: Path,
    prefix: str,
    start_index: int,
    overwrite: bool,
) -> list[Path]:
    template_text = template_path.read_text(encoding="utf-8")

    created: list[Path] = []

    with report_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {report_path}")

        index = start_index
        for row in reader:
            backbone = _required(row, "backbone")
            batch_size = _required(row, "batch_size")
            img_size = _required(row, "img_size")
            seed = (row.get("seed") or "").strip()

            job_name = f"{prefix}{index:03d}"
            out_path = out_dir / f"{job_name}.sh"

            script_text = template_text
            script_text = _replace_flag_once(script_text, "model", backbone)
            script_text = _replace_flag_once(script_text, "batch_size", batch_size)
            script_text = _replace_flag_once(script_text, "img_size", img_size)
            if seed:
                # Only replace if the flag exists in the template
                if _flag_pattern("seed").search(script_text):
                    script_text = _replace_flag_once(script_text, "seed", seed)

            script_text = _replace_log(script_text, f"{job_name}.log")

            if out_path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

            out_path.write_text(script_text, encoding="utf-8", newline="\n")
            created.append(out_path)
            index += 1

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BT PBS job scripts from report.csv")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("script/BT/report.csv"),
        help="Path to report CSV (default: script/BT/report.csv)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("script/BT/BT000.sh"),
        help="Path to template job script (default: script/BT/BT000.sh)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as report)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="BT",
        help="Output file prefix (default: BT)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for numbering (default: 0 => BT000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )

    args = parser.parse_args()

    report_path: Path = args.report
    template_path: Path = args.template

    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    out_dir: Path = args.out_dir if args.out_dir is not None else report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    created = generate_from_report(
        template_path=template_path,
        report_path=report_path,
        out_dir=out_dir,
        prefix=args.prefix,
        start_index=args.start_index,
        overwrite=args.overwrite,
    )

    print(f"Template: {template_path}")
    print(f"Report:   {report_path}")
    print(f"Output:   {out_dir}")
    print(f"Generated {len(created)} scripts:")
    for p in created:
        print(f"- {p.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
