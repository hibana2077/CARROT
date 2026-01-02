#!/usr/bin/env python3
"""Generate PBS job scripts by sweeping carrot_alpha and carrot_topm.

This script takes an existing job script as a template (e.g. script/C/S2/CS2000.sh)
and produces multiple scripts where ONLY these flags are changed:

- --carrot_alpha
- --carrot_topm

It also updates the log filename in the redirection (>> ... 2>&1) to avoid collisions.

Example:
  python script/generate_jobs_from_template.py \
    --template script/C/S2/CS2000.sh
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path


_ALPHA_RE = re.compile(r"(--carrot_alpha\s+)(\d+)(\b)")
_TOPM_RE = re.compile(r"(--carrot_topm\s+)(\d+)(\b)")
_LOG_RE = re.compile(r"(>>\s*)([^\s]+\.log)(\s+2>&1)")


def _replace_flag(script_text: str, pattern: re.Pattern[str], new_value: int, flag_name: str) -> str:
    match = pattern.search(script_text)
    if not match:
        raise ValueError(f"Cannot find {flag_name} in template.")
    return pattern.sub(lambda m: f"{m.group(1)}{new_value}{m.group(3)}", script_text, count=1)


def _replace_log(script_text: str, new_log_name: str) -> str:
    match = _LOG_RE.search(script_text)
    if not match:
        raise ValueError("Cannot find log redirection pattern like '>> something.log 2>&1' in template.")
    return _LOG_RE.sub(lambda m: f"{m.group(1)}{new_log_name}{m.group(3)}", script_text, count=1)


def generate_scripts(
    template_path: Path,
    out_dir: Path,
    alphas: list[int],
    topms: list[int],
    overwrite: bool,
) -> list[Path]:
    template_text = template_path.read_text(encoding="utf-8")

    created: list[Path] = []
    base_name = template_path.stem  # e.g. CS2000

    for alpha, topm in itertools.product(alphas, topms):
        out_stem = f"{base_name}_a{alpha}_m{topm}"
        out_path = out_dir / f"{out_stem}.sh"

        script_text = template_text
        script_text = _replace_flag(script_text, _ALPHA_RE, alpha, "--carrot_alpha")
        script_text = _replace_flag(script_text, _TOPM_RE, topm, "--carrot_topm")
        script_text = _replace_log(script_text, f"{out_stem}.log")

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

        out_path.write_text(script_text, encoding="utf-8", newline="\n")
        created.append(out_path)

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PBS job scripts by scanning alpha/topm combinations.")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("script/C/S2/CS2000.sh"),
        help="Path to template job script (default: script/C/S2/CS2000.sh)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as template)",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="5,10,20",
        help="Comma-separated list of alpha values (default: 5,10,20)",
    )
    parser.add_argument(
        "--topms",
        type=str,
        default="10,20,40",
        help="Comma-separated list of topm values (default: 10,20,40)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )

    args = parser.parse_args()

    template_path: Path = args.template
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    out_dir: Path = args.out_dir if args.out_dir is not None else template_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = [int(x.strip()) for x in args.alphas.split(",") if x.strip()]
    topms = [int(x.strip()) for x in args.topms.split(",") if x.strip()]

    created = generate_scripts(
        template_path=template_path,
        out_dir=out_dir,
        alphas=alphas,
        topms=topms,
        overwrite=args.overwrite,
    )

    print(f"Template: {template_path}")
    print(f"Output dir: {out_dir}")
    print(f"Generated {len(created)} scripts:")
    for p in created:
        print(f"- {p.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
