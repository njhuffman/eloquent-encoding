"""Run the dashboard: ``python -m jepa.dashboard``."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="JEPA training dashboard")
    p.add_argument(
        "--host",
        default=os.environ.get("JEPA_DASHBOARD_HOST", "127.0.0.1"),
        help="Bind address (default: $JEPA_DASHBOARD_HOST or 127.0.0.1; use 0.0.0.0 for Tailscale/LAN)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("JEPA_DASHBOARD_PORT", "8642")),
        help="Port (default: $JEPA_DASHBOARD_PORT or 8642)",
    )
    args = p.parse_args()
    uvicorn.run(
        "jepa.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
