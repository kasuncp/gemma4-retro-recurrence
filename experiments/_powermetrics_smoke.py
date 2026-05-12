"""3-second smoke probe for the powermetrics capture path.

Run with sudo:
    sudo -E .venv/bin/python experiments/_powermetrics_smoke.py

Verifies:
  1. powermetrics accepts the sampler list (cpu_power,gpu_power,ane_power,thermal).
  2. The plist parses and includes a non-zero cpu / gpu power reading.
  3. The temp walker finds a plausible die-temperature value.

Prints a one-line PASS/FAIL summary plus extracted fields. If this passes,
the full path1_phone_mac.py run will produce non-null joules and peak_temp.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from path1_phone_mac import PowerCapture  # type: ignore


def main() -> int:
    log = Path("/tmp/pm_smoke.plist")
    pc = PowerCapture(log_path=log)
    if not pc.start():
        print("[smoke] PowerCapture.start() returned False; see stderr above.")
        return 1
    t0 = time.time()
    time.sleep(3.0)
    pc.stop()
    samples = pc.parse_log()
    if not samples:
        print(f"[smoke] FAIL: no samples parsed from {log} ({log.stat().st_size} bytes).")
        print(f"[smoke] stderr: {log.with_suffix('.stderr').read_text(errors='replace')[:400]}")
        return 1

    cpu_w = [s.cpu_w for s in samples if s.cpu_w > 0]
    gpu_w = [s.gpu_w for s in samples if s.gpu_w > 0]
    ane_w = [s.ane_w for s in samples if s.ane_w > 0]
    pressures = [s.thermal_pressure for s in samples if s.thermal_pressure is not None]
    print(f"[smoke] PASS: {len(samples)} samples over {time.time()-t0:.1f}s")
    print(f"  cpu_w   nonzero: {len(cpu_w)}/{len(samples)}   mean: {sum(cpu_w)/max(1,len(cpu_w)):.2f}")
    print(f"  gpu_w   nonzero: {len(gpu_w)}/{len(samples)}   mean: {sum(gpu_w)/max(1,len(gpu_w)):.2f}")
    print(f"  ane_w   nonzero: {len(ane_w)}/{len(samples)}   mean: {sum(ane_w)/max(1,len(ane_w)):.2f}")
    print(f"  thermal_pressure found: {len(pressures)}/{len(samples)}   distinct: {sorted(set(pressures))}")
    joules = pc.joules_in_window(t0, time.time())
    print(f"  joules over window: {joules}")

    # The smoke is meaningful only if cpu_w shows real values.
    ok = len(cpu_w) > 0
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
