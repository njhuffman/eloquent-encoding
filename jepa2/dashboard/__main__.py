"""Run the jepa2 dashboard: ``python -m jepa2.dashboard``."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="jepa2 training dashboard")
    p.add_argument(
        "--host",
        default=os.environ.get("JEPA2_DASHBOARD_HOST", "127.0.0.1"),
        help="Bind address (default: $JEPA2_DASHBOARD_HOST or 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("JEPA2_DASHBOARD_PORT", "8642")),
        help="Port (default: $JEPA2_DASHBOARD_PORT or 8642, to avoid clashing with jepa on 8642)",
    )
    args = p.parse_args()
    uvicorn.run(
        "jepa2.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
