#!/usr/bin/env python3
"""Generate PBS job scripts by sweeping parameter combinations.

This script takes an existing job script as a template (e.g. script/C/S2/CS2000.sh)
and produces multiple scripts where ONLY the specified flags are changed.

You can sweep any flags by repeating `--sweep`:
    --sweep carrot_alpha=5,10,20 --sweep carrot_topm=10,20,40

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


_LOG_RE = re.compile(r"(>>\s*)([^\s]+\.log)(\s+2>&1)")


def _flag_pattern(flag: str) -> re.Pattern[str]:
    # Matches: --flag <value>
    # Value can be int or float (and scientific notation) since templates may vary.
    return re.compile(rf"(--{re.escape(flag)}\s+)([^\s\\]+)(\b)")


def _replace_flag(script_text: str, flag: str, new_value: str) -> str:
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


def generate_scripts(
    template_path: Path,
    out_dir: Path,
    sweeps: list[tuple[str, list[str]]],
    overwrite: bool,
) -> list[Path]:
    template_text = template_path.read_text(encoding="utf-8")

    created: list[Path] = []
    base_name = template_path.stem  # e.g. CS2000

    sweep_flags = [name for name, _values in sweeps]
    sweep_values = [_values for _name, _values in sweeps]

    for combo in itertools.product(*sweep_values):
        suffix = "_".join(f"{flag}{val}" for flag, val in zip(sweep_flags, combo))
        out_stem = f"{base_name}_{suffix}" if suffix else base_name
        out_path = out_dir / f"{out_stem}.sh"

        script_text = template_text
        for flag, val in zip(sweep_flags, combo):
            script_text = _replace_flag(script_text, flag, val)
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
        "--sweep",
        action="append",
        default=None,
        help=(
            "Repeatable. Format: flag=v1,v2,... (flag name WITHOUT leading --). "
            "Example: --sweep carrot_alpha=5,10,20 --sweep carrot_topm=10,20,40"
        ),
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

    if args.sweep is None:
        # Backward-compatible default behavior
        sweeps: list[tuple[str, list[str]]] = [
            ("carrot_alpha", ["5", "10", "20"]),
            ("carrot_topm", ["10", "20", "40"]),
        ]
    else:
        sweeps = []
        for item in args.sweep:
            if "=" not in item:
                raise ValueError(f"Invalid --sweep '{item}'. Expected flag=v1,v2,...")
            flag, values_csv = item.split("=", 1)
            flag = flag.strip()
            if not flag:
                raise ValueError(f"Invalid --sweep '{item}': empty flag")
            values = [v.strip() for v in values_csv.split(",") if v.strip()]
            if not values:
                raise ValueError(f"Invalid --sweep '{item}': no values")
            sweeps.append((flag, values))

    created = generate_scripts(
        template_path=template_path,
        out_dir=out_dir,
        sweeps=sweeps,
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
