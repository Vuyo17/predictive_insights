from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
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
    port: int


def password_env_key(node_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in node_name.upper())
    return f"CLUSTER_SSH_PASSWORD_{slug}"


def load_nodes(config_path: Path) -> List[NodeConfig]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    items = payload.get("nodes", []) if isinstance(payload, dict) else []
    if not items:
        raise ValueError("No nodes found in config")

    nodes: List[NodeConfig] = []
    for item in items:
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
                port=int(item.get("port", 5173)),
            )
        )
    return nodes


def resolve_passwords(nodes: List[NodeConfig]) -> Dict[str, str]:
    passwords: Dict[str, str] = {}
    for node in nodes:
        env_key = password_env_key(node.name)
        env_value = os.environ.get(env_key)
        if env_value:
            passwords[node.name] = env_value
        else:
            passwords[node.name] = getpass.getpass(f"Password for {node.user}@{node.host} ({node.name}): ")
    return passwords


def connect(node: NodeConfig, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=node.host,
        username=node.user,
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=20,
    )
    return client


def run_remote(client: paramiko.SSHClient, command: str) -> str:
    _, stdout, _ = client.exec_command(command)
    _ = stdout.channel.recv_exit_status()
    return stdout.read().decode("utf-8", errors="replace").strip()


def remote_project_expr(project_dir: str) -> str:
    if project_dir == "~":
        return "$HOME"
    if project_dir.startswith("~/"):
        return "$HOME/" + project_dir[2:]
    return project_dir


def pull_logs_for_node(node: NodeConfig, password: str, log_lines: int) -> Dict[str, object]:
    project_expr = remote_project_expr(node.project_dir)
    client = connect(node, password)
    try:
        backend_proc_cmd = (
            f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR\"; "
            "pgrep -u \"$USER\" -af \"[c]ontinuous_improve_pipeline.py\" || true"
        )
        backend_log_cmd = (
            f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR\"; "
            f"if [ -f logs/backend_{node.name}.log ]; then tail -n {log_lines} logs/backend_{node.name}.log; "
            "elif [ -f logs/backend.log ]; then tail -n 40 logs/backend.log; "
            "else echo '(no backend logs yet)'; fi"
        )
        backend_state_cmd = (
            f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR\"; "
            f"if [ -f outputs/continuous_state_{node.name}.json ]; then cat outputs/continuous_state_{node.name}.json; "
            "elif [ \""
            + node.name
            + "\" = \"primary\" ] && [ -f outputs/continuous_state.json ]; then cat outputs/continuous_state.json; "
            "else echo '{}'; fi"
        )

        backend_procs = [line for line in run_remote(client, backend_proc_cmd).splitlines() if line.strip()]
        backend_log = run_remote(client, backend_log_cmd)
        backend_state_raw = run_remote(client, backend_state_cmd)

        backend_state: Dict[str, object] = {}
        try:
            parsed = json.loads(backend_state_raw)
            if isinstance(parsed, dict):
                backend_state = {
                    "status": parsed.get("status"),
                    "updated_at": parsed.get("updated_at"),
                    "current_best_cv_auc": parsed.get("current_best_cv_auc"),
                    "iterations_completed": parsed.get("iterations_completed"),
                }
        except json.JSONDecodeError:
            backend_state = {}

        payload: Dict[str, object] = {
            "name": node.name,
            "host": node.host,
            "user": node.user,
            "roles": {
                "frontend": node.frontend,
                "backend": node.backend,
            },
            "backend": {
                "running": len(backend_procs) > 0,
                "processes": backend_procs,
                "logTail": backend_log,
                "state": backend_state,
            },
        }

        if node.frontend:
            frontend_proc_pattern = f"[v]ite|[h]ttp.server {int(node.port)}"
            frontend_proc_cmd = (
                f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR\"; "
                f"pgrep -u \"$USER\" -af \"{frontend_proc_pattern}\" || true"
            )
            frontend_log_cmd = (
                f"PROJECT_DIR={project_expr}; cd \"$PROJECT_DIR\"; "
                f"if [ -f logs/frontend_{node.name}.log ]; then tail -n {log_lines} logs/frontend_{node.name}.log; "
                "elif [ -f logs/frontend.log ]; then tail -n 40 logs/frontend.log; "
                "else echo '(no frontend logs yet)'; fi"
            )
            frontend_procs = [line for line in run_remote(client, frontend_proc_cmd).splitlines() if line.strip()]
            frontend_log = run_remote(client, frontend_log_cmd)
            payload["frontend"] = {
                "running": len(frontend_procs) > 0,
                "processes": frontend_procs,
                "logTail": frontend_log,
                "url": f"http://{node.host}:{node.port}",
            }

        return payload
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull fresh runtime logs from all cluster servers.")
    parser.add_argument("--config", type=Path, default=Path("ops/cluster_nodes.json"))
    parser.add_argument("--out", type=Path, default=Path("frontend/src/data/server_logs.json"))
    parser.add_argument("--log-lines", type=int, default=80)
    args = parser.parse_args()

    nodes = load_nodes(args.config)
    passwords = resolve_passwords(nodes)

    servers = [pull_logs_for_node(node, passwords[node.name], args.log_lines) for node in nodes]
    result = {
        "generatedAt": dt.datetime.now().isoformat(),
        "servers": servers,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote {args.out} with {len(servers)} server entries")


if __name__ == "__main__":
    main()
