from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import paramiko

CONFIG_PATH = Path("ops/cluster_nodes.json")


def password_env_key(node_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in node_name.upper())
    return f"CLUSTER_SSH_PASSWORD_{slug}"


def main() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    primary = next((n for n in nodes if bool(n.get("frontend", False))), None)
    if not primary:
        raise SystemExit("No frontend node configured")

    name = str(primary["name"])
    host = str(primary["host"])
    user = str(primary["user"])
    project_dir = str(primary["project_dir"])
    port = int(primary.get("port", 5173))

    env_key = password_env_key(name)
    password = os.environ.get(env_key) or getpass.getpass(f"Password for {user}@{host}: ")

    project_expr = "$HOME/" + project_dir[2:] if project_dir.startswith("~/") else project_dir

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=password, look_for_keys=False, allow_agent=False, timeout=20)
    try:
        commands = [
            (
                "check_dist",
                f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR/frontend\"; "
                "if [ -d dist ]; then echo DIST_OK; else echo DIST_MISSING; fi",
            ),
            (
                "stop_existing",
                f"pkill -f \"[v]ite\" >/dev/null 2>&1 || true; "
                f"pkill -f \"[h]ttp.server {port}\" >/dev/null 2>&1 || true; echo STOP_OK",
            ),
            (
                "start_static",
                f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR/frontend\"; "
                "PY_FALLBACK=python3; "
                "if command -v python >/dev/null 2>&1; then PY_FALLBACK=python; fi; "
                f"nohup env PYTHONUNBUFFERED=1 $PY_FALLBACK -m http.server {port} --directory dist > ../logs/frontend_{name}.log 2>&1 < /dev/null & "
                "sleep 1; echo START_OK",
            ),
            (
                "check_proc",
                f"pgrep -u \"$USER\" -af \"[h]ttp.server {port}\" || true",
            ),
            (
                "tail_log",
                f"tail -n 40 ~/predictive_insights/logs/frontend_{name}.log 2>/dev/null || true",
            ),
        ]

        for label, command in commands:
            _, stdout, stderr = client.exec_command(command)
            code = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="replace").strip()
            err = stderr.read().decode("utf-8", errors="replace").strip()
            print(f"[{label}] exit={code}")
            if out:
                print(out)
            if err:
                print(err)

        print(f"Started static frontend on {host}:{port}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
