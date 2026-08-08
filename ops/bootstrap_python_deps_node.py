from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

import paramiko


def password_env_key(node_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in node_name.upper())
    return f"CLUSTER_SSH_PASSWORD_{slug}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Python deps on one cluster node")
    parser.add_argument("--config", type=Path, default=Path("ops/cluster_nodes.json"))
    parser.add_argument("--node", type=str, required=True)
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    node = next((n for n in nodes if n.get("name") == args.node), None)
    if not node:
        raise SystemExit(f"Node not found: {args.node}")

    name = str(node["name"])
    host = str(node["host"])
    user = str(node["user"])
    project_dir = str(node["project_dir"])
    python_bin = str(node.get("python_bin", "python3"))

    env_key = password_env_key(name)
    password = os.environ.get(env_key) or getpass.getpass(f"Password for {user}@{host} ({name}): ")

    if project_dir.startswith("~/"):
        project_expr = "$HOME/" + project_dir[2:]
    elif project_dir == "~":
        project_expr = "$HOME"
    else:
        project_expr = project_dir

    cmd = (
        f"PROJECT_DIR={project_expr}; "
        f"cd \"$PROJECT_DIR\"; "
        f"if ! {python_bin} -m pip --version >/dev/null 2>&1; then "
        "if command -v curl >/dev/null 2>&1; then "
        "curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; "
        "elif command -v wget >/dev/null 2>&1; then "
        "wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py; "
        "else echo 'Missing curl/wget for pip bootstrap' >&2; exit 1; fi; "
        f"{python_bin} /tmp/get-pip.py --user --break-system-packages; "
        "fi; "
        f"{python_bin} -m pip install --user --break-system-packages -r requirements.txt"
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=password, look_for_keys=False, allow_agent=False, timeout=20)
    try:
        _, stdout, stderr = client.exec_command(cmd)
        code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print(out)
        if err.strip():
            print(err)
        if code != 0:
            raise SystemExit(code)
        print(f"Python dependencies installed on {name}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
