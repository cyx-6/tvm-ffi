#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Parse results/*.log from run_bench.sh into a main|head-off|head-on table.

For each (op, python) we print main / head-off / head-on (min sec/call), plus
derived deltas:
  Dlayout = head-off - main      (PR machinery present but inert)
  Dtying  = head-on  - head-off  (tying logic, layout held constant)
The free-threaded spin-lock tax for an op = Dtying(3.14t) - Dtying(3.13).
Memory rows (bytes_per_obj) are reported separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

# label -> (config, python). config in {main, headoff, headon}.
PYS = ["3.12", "3.13", "3.14t"]
LABELS = {}
for _py in PYS:
    LABELS[f"main-{_py}"] = ("main", _py)
    LABELS[f"headoff-{_py}"] = ("headoff", _py)
    LABELS[f"headon-{_py}"] = ("headon", _py)


def parse_log(path: Path) -> dict[tuple[str, str], float]:
    """Return {(op, stat): value} from RESULT lines."""
    out: dict[tuple[str, str], float] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("RESULT\t"):
            continue
        _, op, stat, val = line.split("\t")
        out[(op, stat)] = float(val)
    return out


def fmt_ns(sec: float | None) -> str:
    if sec is None:
        return "    --   "
    return f"{sec * 1e9:8.1f}"


def fmt_pct(delta: float | None, base: float | None) -> str:
    if delta is None or base is None or base == 0:
        return "   -- "
    return f"{100.0 * delta / base:+6.1f}%"


def main() -> None:
    res_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    data: dict[str, dict[tuple[str, str], float]] = {}
    for label in LABELS:
        p = res_dir / f"{label}.log"
        if p.exists():
            data[label] = parse_log(p)
        else:
            print(f"# missing log: {p}", file=sys.stderr)

    # Collect ops with a "min" stat (latency) vs "bytes_per_obj" (memory).
    lat_ops: list[str] = []
    mem_ops: list[str] = []
    seen = set()
    for d in data.values():
        for (op, stat) in d:
            if op in seen:
                continue
            if stat == "min":
                lat_ops.append(op)
                seen.add(op)
            elif stat == "bytes_per_obj":
                mem_ops.append(op)
                seen.add(op)

    def get(config: str, py: str, op: str, stat: str) -> float | None:
        for label, (c, p) in LABELS.items():
            if c == config and p == py and label in data:
                return data[label].get((op, stat))
        return None

    # A cell is "noisy" if the FLOOR was barely sampled: p20 (3rd-smallest of K) sits far
    # above min. This (unlike median-min) ignores a slow tail when min-of-K still found a
    # clean floor across several trials. Guarded by an absolute-ns floor so sub-ns jitter on
    # a ~5ns op can't trip the relative test.
    NOISE_REL = 0.15  # p20 more than 15% above min
    NOISE_ABS_NS = 1.0  # and at least 1ns above min

    def noise(config: str, py: str, op: str) -> float | None:
        # Adaptive ops (C++ microbenches) emit an authoritative converged flag: if the floor
        # was certified this run, trust it (score 0); if not, flag it (score 2).
        conv = get(config, py, op, "converged")
        if conv is not None:
            return 0.0 if conv >= 1.0 else 2.0
        # Python ops: floor-dispersion heuristic. Flag only if p20 sits >NOISE_REL AND
        # >NOISE_ABS_NS above min (so sub-ns jitter on a fast op can't trip it).
        mn = get(config, py, op, "min")
        p20 = get(config, py, op, "p20") or get(config, py, op, "median")
        if mn is None or mn == 0 or p20 is None:
            return None
        gap = p20 - mn
        return min(gap / mn / NOISE_REL, gap * 1e9 / NOISE_ABS_NS)  # both must clear 1.0

    for py in PYS:
        print(f"\n### Python {py}  (latency, ns/call, min-of-K)   "
              f"[* = floor not certified (cpp: adaptive converged=0; py: p20>min by "
              f">{NOISE_REL:.0%} & >{NOISE_ABS_NS:.0f}ns)]")
        hdr = (
            f"{'op':<26} {'main':>9} {'head-off':>9} {'head-on':>9}  "
            f"{'Dlayout':>8} {'layout%':>7}  {'Dtying':>8} {'tying%':>7}  "
            f"{'Dtotal':>8} {'total%':>7} N"
        )
        print(hdr)
        print("-" * len(hdr))
        for op in lat_ops:
            m = get("main", py, op, "min")
            off = get("headoff", py, op, "min")
            on = get("headon", py, op, "min")
            dlayout = (off - m) if (off is not None and m is not None) else None
            dtying = (on - off) if (on is not None and off is not None) else None
            dtotal = (on - m) if (on is not None and m is not None) else None  # end-to-end
            noises = [noise(c, py, op) for c in ("main", "headoff", "headon")]
            noisy = any(x is not None and x > 1.0 for x in noises)
            print(
                f"{op:<26} {fmt_ns(m)} {fmt_ns(off)} {fmt_ns(on)}  "
                f"{fmt_ns(dlayout)} {fmt_pct(dlayout, m)}  {fmt_ns(dtying)} {fmt_pct(dtying, off)}  "
                f"{fmt_ns(dtotal)} {fmt_pct(dtotal, m)} {'*' if noisy else ' '}"
            )

    if mem_ops:
        print("\n### Memory (bytes/object, RSS delta)")
        hdr = f"{'op':<20} {'py':<6} {'main':>9} {'head-off':>9} {'head-on':>9}  {'on-off':>8}"
        print(hdr)
        print("-" * len(hdr))
        for py in PYS:
            for op in mem_ops:
                m = get("main", py, op, "bytes_per_obj")
                off = get("headoff", py, op, "bytes_per_obj")
                on = get("headon", py, op, "bytes_per_obj")
                d = (on - off) if (on is not None and off is not None) else None
                row = (
                    f"{op:<20} {py:<6} "
                    f"{('--' if m is None else f'{m:9.1f}')} "
                    f"{('--' if off is None else f'{off:9.1f}')} "
                    f"{('--' if on is None else f'{on:9.1f}')}  "
                    f"{('--' if d is None else f'{d:8.1f}')}"
                )
                print(row)

    # FT spin-lock tax: Dtying(3.14t) - Dtying(3.13) per op.
    print("\n### Free-threaded spin-lock tax  (Dtying[3.14t] - Dtying[3.13], ns/call)")
    print(f"{'op':<26} {'tax':>9}")
    print("-" * 36)
    for op in lat_ops:
        on_ft, off_ft = get("headon", "3.14t", op, "min"), get("headoff", "3.14t", op, "min")
        on_g, off_g = get("headon", "3.13", op, "min"), get("headoff", "3.13", op, "min")
        if None in (on_ft, off_ft, on_g, off_g):
            continue
        tax = (on_ft - off_ft) - (on_g - off_g)
        print(f"{op:<26} {tax * 1e9:9.1f}")


if __name__ == "__main__":
    main()
