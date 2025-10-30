import os, json, hashlib, time, platform, getpass
from typing import Dict, Any

def write_manifest(out_path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = dict(payload)
    payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload["platform"] = platform.platform()
    payload["user"] = getpass.getuser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
