#!/usr/bin/env python3
"""Rubik Pi 3 sampler: CPU clocks (3 domains) + SoC thermals at ~20 Hz.

The QCM6490 board has no trustworthy input-power sensor (the qcom-battmgr
power_supply entries return inconsistent values), so this logger records
clocks and temperatures only.
Columns: unix_t, cpu0_mhz, cpu4_mhz, cpu7_mhz, soc_temp_c
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

POLICIES = [0, 4, 7]
THERMAL_TYPES = ("cpuss0-thermal", "cpuss1-thermal", "nspss0-thermal",
                 "nspss1-thermal", "gpuss0-thermal", "gpuss1-thermal")


def read_mhz(policy: int) -> float:
    try:
        return int(Path(f"/sys/devices/system/cpu/cpufreq/policy{policy}/scaling_cur_freq")
                   .read_text()) / 1000.0
    except OSError:
        return float("nan")


def read_thermals() -> dict[str, float]:
    out = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name = (zone / "type").read_text().strip()
            if name in THERMAL_TYPES:
                out[name] = int((zone / "temp").read_text()) / 1000.0
        except (OSError, ValueError):
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["unix_t", "cpu0_mhz", "cpu4_mhz", "cpu7_mhz", "soc_temp_c"])
        fh.flush()
        while True:
            temps = read_thermals()
            writer.writerow([f"{time.time():.3f}",
                             *(f"{read_mhz(p):.0f}" for p in POLICIES),
                             f"{max(temps.values(), default=float('nan')):.1f}"])
            fh.flush()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
