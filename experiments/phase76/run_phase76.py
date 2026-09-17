import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase76.reports.generate_report import generate_phase76_full_report

def main():
    print("=" * 65)
    print("  PHASE 76 — CONTEXT & LOSS CALIBRATION EXPERIMENT")
    print("=" * 65)
    generate_phase76_full_report()
    print("=" * 65)
    print("  STATUS: PHASE 76 EXECUTION & REPORTING COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    main()
