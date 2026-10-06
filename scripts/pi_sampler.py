"""Pi 5 sampler: CPU clock + SoC temperature at ~10 Hz, PMIC rail power at ~5 Hz.

Pi 5 has no board-input sensor. The PMIC (DA9091) reports per-rail voltage and
current on the DC-DC output side; we log the sum (SoC + DDR + on-board rails).
The Hailo-8 on the M.2 HAT+ draws from the 5 V GPIO rail, which the PMIC does
not instrument, so its power is NOT included.
CSV: unix_t, cpu_mhz, temp_c, pmic_w (pmic_w repeated between its own samples).
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

POLICY0 = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
RAIL = re.compile(r"^\s*(\S+)_(A|V) .*\((\d+)\)=([0-9.]+)[AV]")


def pmic_watts() -> float:
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True,
                         text=True).stdout
    rails = {}
    for line in out.splitlines():
        m = RAIL.match(line)
        if m:
            name, kind, _, value = m.groups()
            rails.setdefault(name, {})[kind] = float(value)
    return sum(v["A"] * v["V"] for v in rails.values() if "A" in v and "V" in v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()
    period = 1.0 / args.hz
    watts = float("nan")
    with args.out.open("w") as fh:
        i = 0
        while True:
            t0 = time.perf_counter()
            if i % 2 == 0:
                watts = pmic_watts()
            i += 1
            mhz = int(POLICY0.read_text()) / 1000
            temp = int(THERMAL.read_text()) / 1000
            fh.write(f"{time.time():.4f},{mhz:.0f},{temp:.1f},{watts:.3f}\n")
            fh.flush()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
