from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_sgbikd._legacy import import_root_module


def main():
    import_root_module("trafficgaze_w3da_yolo12s_bikd_metrics").main()


if __name__ == "__main__":
    main()

