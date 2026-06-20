"""Run the jepa3 dashboard: ``python -m jepa3.dashboard``."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="jepa3 training dashboard")
    p.add_argument(
        "--host",
        default=os.environ.get("JEPA3_DASHBOARD_HOST", "127.0.0.1"),
        help="Bind address (default: $JEPA3_DASHBOARD_HOST or 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("JEPA3_DASHBOARD_PORT", "8767")),
        help="Port (default: $JEPA3_DASHBOARD_PORT or 8767)",
    )
    args = p.parse_args()
    uvicorn.run(
        "jepa3.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
