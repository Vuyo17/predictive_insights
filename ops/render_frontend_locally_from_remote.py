from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import paramiko


@dataclass
class NodeConfig:
    name: str
    host: str
    user: str
    project_dir: str
    frontend: bool
    backend: bool


def password_env_key(node_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in node_name.upper())
    return f"CLUSTER_SSH_PASSWORD_{slug}"


def load_nodes(config_path: Path) -> List[NodeConfig]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    nodes: List[NodeConfig] = []

    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        nodes.append(
            NodeConfig(
                name=str(item["name"]),
                host=str(item["host"]),
                user=str(item["user"]),
                project_dir=str(item["project_dir"]),
                frontend=bool(item.get("frontend", False)),
                backend=bool(item.get("backend", True)),
            )
        )

    if not nodes:
        raise ValueError("No nodes found in cluster config")

    return nodes


def resolve_passwords(nodes: List[NodeConfig]) -> Dict[str, str]:
    passwords: Dict[str, str] = {}
    for node in nodes:
        env_key = password_env_key(node.name)
        env_value = os.environ.get(env_key)
        if env_value:
            passwords[node.name] = env_value
            continue
        prompt = f"Password for {node.user}@{node.host} ({node.name}): "
        passwords[node.name] = getpass.getpass(prompt)
    return passwords


def remote_path(project_dir: str) -> str:
    if project_dir == "~":
        return "$HOME"
    if project_dir.startswith("~/"):
        return "$HOME/" + project_dir[2:]
    return project_dir


def connect(node: NodeConfig, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=node.host,
        username=node.user,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
    )
    return client


def list_remote_output_files(client: paramiko.SSHClient, node: NodeConfig) -> List[str]:
    project_expr = remote_path(node.project_dir)
    cmd = (
        f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR\"; "
        "ls outputs 2>/dev/null || true"
    )
    _, stdout, _ = client.exec_command(cmd)
    _ = stdout.channel.recv_exit_status()
    names = [line.strip() for line in stdout.read().decode("utf-8", errors="replace").splitlines() if line.strip()]
    return names


def resolve_remote_project_dir(client: paramiko.SSHClient, node: NodeConfig) -> str:
    project_expr = remote_path(node.project_dir)
    cmd = (
        f"PROJECT_DIR={project_expr}; "
        "cd \"$PROJECT_DIR\" 2>/dev/null && pwd || true"
    )
    _, stdout, _ = client.exec_command(cmd)
    _ = stdout.channel.recv_exit_status()
    resolved = stdout.read().decode("utf-8", errors="replace").strip()
    if not resolved:
        raise RuntimeError(f"Could not resolve remote project dir for node {node.name}")
    return resolved


def should_copy_output_file(file_name: str) -> bool:
    patterns = [
        r"^submission.*\.csv$",
        r"^continuous_report_.*\.json$",
        r"^auto_report_.*\.json$",
        r"^cv_metrics\.json$",
        r"^continuous_state.*\.json$",
    ]
    return any(re.match(pattern, file_name, flags=re.IGNORECASE) for pattern in patterns)


def copy_remote_outputs_for_node(
    client: paramiko.SSHClient,
    node: NodeConfig,
    local_outputs_dir: Path,
) -> int:
    names = list_remote_output_files(client, node)
    to_copy = [name for name in names if should_copy_output_file(name)]
    copied = 0
    remote_project_dir = resolve_remote_project_dir(client, node)

    sftp = client.open_sftp()
    try:
        for name in to_copy:
            remote_file = f"{remote_project_dir}/outputs/{name}"
            local_name = name
            local_path = local_outputs_dir / local_name

            if local_path.exists():
                if node.name == "primary":
                    pass
                else:
                    local_name = f"{node.name}__{name}"
                    local_path = local_outputs_dir / local_name

            try:
                sftp.get(remote_file, str(local_path))
                copied += 1
            except FileNotFoundError:
                continue
    finally:
        sftp.close()

    return copied


def run_refresh(frontend_dir: Path) -> None:
    subprocess.run(
        ["node", "scripts/build-submission-metrics.mjs"],
        cwd=str(frontend_dir),
        check=True,
    )


def run_server_logs_pull(project_root: Path) -> None:
    subprocess.run(
        [
            "py",
            "ops/pull_server_runtime_logs.py",
            "--config",
            "ops/cluster_nodes.json",
            "--out",
            "frontend/src/data/server_logs.json",
            "--log-lines",
            "80",
        ],
        cwd=str(project_root),
        check=True,
    )


def run_local_frontend(frontend_dir: Path, port: int) -> None:
    subprocess.run(
        ["npm.cmd", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(frontend_dir),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull remote backend artifacts over SSH and render frontend locally."
    )
    parser.add_argument("--config", type=Path, default=Path("ops/cluster_nodes.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--serve", action="store_true", help="Start local frontend dev server after refresh")
    parser.add_argument("--port", type=int, default=5173, help="Local frontend port when using --serve")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    frontend_dir = project_root / "frontend"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(args.config)
    passwords = resolve_passwords(nodes)

    total_copied = 0
    for node in nodes:
        if not node.backend:
            continue
        print(f"Connecting to {node.name} ({node.user}@{node.host})...")
        client = connect(node, passwords[node.name])
        try:
            copied = copy_remote_outputs_for_node(client, node, outputs_dir)
            total_copied += copied
            print(f"  Copied {copied} output file(s)")
        finally:
            client.close()

    print(f"Total copied files: {total_copied}")

    run_server_logs_pull(project_root)
    run_refresh(frontend_dir)

    print("Local frontend data refreshed from remote servers.")
    print("Open: http://localhost:{0}".format(args.port))

    if args.serve:
        run_local_frontend(frontend_dir, args.port)


if __name__ == "__main__":
    main()
