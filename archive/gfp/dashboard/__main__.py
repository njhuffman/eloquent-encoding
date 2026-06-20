"""Run the gfp dashboard: ``python -m gfp.dashboard``."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="gfp training dashboard")
    p.add_argument(
        "--host",
        default=os.environ.get("GFP_DASHBOARD_HOST", "127.0.0.1"),
        help="Bind address (default: $GFP_DASHBOARD_HOST or 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GFP_DASHBOARD_PORT", "8768")),
        help="Port (default: $GFP_DASHBOARD_PORT or 8768)",
    )
    args = p.parse_args()
    uvicorn.run(
        "gfp.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
