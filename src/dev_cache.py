import json
from datetime import date
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".dev_cache"


def _cache_path(key: str) -> Path:
    today = date.today().isoformat()
    return CACHE_DIR / today / f"{key}.json"


def save_cache(key: str, data) -> Path:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_cache(key: str):
    path = _cache_path(key)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
