"""Удобный локальный запуск: ``python site/run.py [--reload] [--port 8000]`` из любого каталога."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="run the team website")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    if str(SITE_DIR) not in sys.path:
        sys.path.insert(0, str(SITE_DIR))
    import uvicorn  # noqa: WPS433

    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload,
                app_dir=str(SITE_DIR), workers=1, proxy_headers=True)


if __name__ == "__main__":
    main()
