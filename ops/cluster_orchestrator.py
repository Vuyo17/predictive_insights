from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import paramiko
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'paramiko'. Install with: pip install paramiko"
    ) from exc


@dataclass
class NodeConfig:
    name: str
    host: str
    user: str
    project_dir: str
    frontend: bool = False
    backend: bool = True
    python_bin: str = "python3"
    npm_bin: str = "npm"
    port: int = 5173


def load_nodes(config_path: Path) -> List[NodeConfig]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "nodes" not in payload:
        raise ValueError("Config must be a JSON object with a 'nodes' array")

    nodes_raw = payload.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("Config must include at least one node in 'nodes'")

    nodes: List[NodeConfig] = []
    for item in nodes_raw:
        if not isinstance(item, dict):
            raise ValueError("Each node entry must be an object")

        node = NodeConfig(
            name=str(item["name"]),
            host=str(item["host"]),
            user=str(item["user"]),
            project_dir=str(item["project_dir"]),
            frontend=bool(item.get("frontend", False)),
            backend=bool(item.get("backend", True)),
            python_bin=str(item.get("python_bin", "python3")),
            npm_bin=str(item.get("npm_bin", "npm")),
            port=int(item.get("port", 5173)),
        )
        nodes.append(node)

    if sum(1 for n in nodes if n.frontend) != 1:
        raise ValueError("Exactly one node must have frontend=true")

    if not any(n.backend for n in nodes):
        raise ValueError("At least one node must have backend=true")

    return nodes


def password_env_key(node_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in node_name.upper())
    return f"CLUSTER_SSH_PASSWORD_{slug}"


def resolve_passwords(nodes: List[NodeConfig]) -> Dict[str, str]:
    passwords: Dict[str, str] = {}

    for node in nodes:
        env_key = password_env_key(node.name)
        value = os.environ.get(env_key)
        if value:
            passwords[node.name] = value
            continue

        prompt = f"Password for {node.user}@{node.host} ({node.name}): "
        passwords[node.name] = getpass.getpass(prompt)

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
        timeout=25,
    )
    return client


def run_remote(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, out, err


def sh(value: str) -> str:
    return shlex.quote(value)


def remote_path_expr(remote_path: str) -> str:
    if remote_path == "~":
        return "$HOME"
    if remote_path.startswith("~/"):
        return "$HOME/" + sh(remote_path[2:])
    return sh(remote_path)


def create_project_archive(local_project_dir: Path, archive_path: Path) -> None:
    excluded_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    excluded_file_suffixes = {".pyc", ".pyo"}

    with tarfile.open(archive_path, mode="w:gz") as tar:
        for root, dirs, files in os.walk(local_project_dir):
            rel_root = Path(root).relative_to(local_project_dir)
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file_name in files:
                file_path = Path(root) / file_name
                if file_path.suffix.lower() in excluded_file_suffixes:
                    continue
                if file_name in {".DS_Store"}:
                    continue

                arcname = (rel_root / file_name).as_posix()
                tar.add(file_path, arcname=arcname)


def sync_project_to_node(
    node: NodeConfig,
    password: str,
    archive_path: Path,
) -> None:
    client = connect(node, password)
    try:
        project_expr = remote_path_expr(node.project_dir)
        prepare_cmd = "mkdir -p $HOME/tmp"
        code, out, err = run_remote(client, prepare_cmd)
        if code != 0:
            raise RuntimeError(f"Unable to prepare tmp dir: {err.strip() or out.strip()}")

        sftp = client.open_sftp()
        try:
            remote_bundle = "tmp/predictive_insights_bundle.tar.gz"
            sftp.put(str(archive_path), remote_bundle)
        finally:
            sftp.close()

        deploy_cmd = (
            f"PROJECT_DIR={project_expr} && "
            f"mkdir -p \"$PROJECT_DIR\" && "
            f"tar -xzf $HOME/tmp/predictive_insights_bundle.tar.gz -C \"$PROJECT_DIR\""
        )
        code, out, err = run_remote(client, deploy_cmd)
        if code != 0:
            raise RuntimeError(f"Deploy extract failed: {err.strip() or out.strip()}")
    finally:
        client.close()


def sync_cluster(nodes: List[NodeConfig], passwords: Dict[str, str], args: argparse.Namespace) -> None:
    local_project_dir = Path(args.local_project_dir).resolve()
    if not local_project_dir.exists():
        raise ValueError(f"Local project dir not found: {local_project_dir}")

    with tempfile.TemporaryDirectory(prefix="predictive_sync_") as tmp_dir:
        archive_path = Path(tmp_dir) / "predictive_insights_bundle.tar.gz"
        create_project_archive(local_project_dir, archive_path)

        for node in nodes:
            print(f"\n[{node.name}] Syncing project to {node.user}@{node.host}...")
            sync_project_to_node(node, passwords[node.name], archive_path)
            print(f"[{node.name}] sync complete")


def build_backend_command(
    node: NodeConfig,
    shard_index: int,
    shard_count: int,
    args: argparse.Namespace,
) -> str:
    project_expr = remote_path_expr(node.project_dir)
    train_file = sh(args.train_file)
    test_file = sh(args.test_file)
    scores_file = sh(args.leaderboard_scores_file)

    worker_name = sh(node.name)
    return (
        f"PROJECT_DIR={project_expr} && "
        f"cd \"$PROJECT_DIR\" && "
        f"mkdir -p logs outputs && "
        f"PY_EXEC=\"{node.python_bin}\" && "
        f"if [ -x .venv/bin/python ] && .venv/bin/python -c \"import numpy\" >/dev/null 2>&1; then PY_EXEC=.venv/bin/python; fi && "
        f"nohup env PYTHONUNBUFFERED=1 $PY_EXEC src/continuous_improve_pipeline.py "
        f"--train-file {train_file} "
        f"--test-file {test_file} "
        f"--n-splits {args.n_splits} "
        f"--confirm-splits {args.confirm_splits} "
        f"--batch-size {args.batch_size} "
        f"--parallel-jobs {args.parallel_jobs} "
        f"--min-improvement {args.min_improvement} "
        f"--seed {args.seed} "
        f"--server-count {shard_count} "
        f"--server-index {shard_index} "
        f"--worker-name {worker_name} "
        f"--leaderboard-scores-file {scores_file} "
        f"--output-dir outputs "
        f"> logs/backend_{node.name}.log 2>&1 < /dev/null &"
    )


def build_frontend_command(node: NodeConfig) -> str:
    project_expr = remote_path_expr(node.project_dir)
    port = int(node.port)
    npm = sh(node.npm_bin)
    return (
        f"PROJECT_DIR={project_expr} && "
        f"cd \"$PROJECT_DIR/frontend\" && "
        f"mkdir -p ../logs && "
        "pkill -f \"[v]ite\" >/dev/null 2>&1 || true && "
        f"pkill -f \"[h]ttp.server {port}\" >/dev/null 2>&1 || true && "
        f"NPM_BIN=$(command -v {npm} 2>/dev/null || true) && "
        "if [ -n \"$NPM_BIN\" ] && [ -x \"$NPM_BIN\" ]; then "
        f"nohup \"$NPM_BIN\" run dev -- --host 0.0.0.0 --port {port} > ../logs/frontend_{node.name}.log 2>&1 < /dev/null & "
        "else "
        "PY_FALLBACK=python3 && "
        "if command -v python >/dev/null 2>&1; then PY_FALLBACK=python; fi && "
        "if [ ! -d dist ]; then echo 'Frontend dist missing and npm unavailable' > ../logs/frontend_"
        + node.name
        + ".log; exit 1; fi && "
        f"nohup env PYTHONUNBUFFERED=1 $PY_FALLBACK -m http.server {port} --directory dist > ../logs/frontend_{node.name}.log 2>&1 < /dev/null & "
        "fi"
    )


def build_bootstrap_command(node: NodeConfig, repo_url: Optional[str], branch: str) -> str:
    project_expr = remote_path_expr(node.project_dir)
    py = sh(node.python_bin)
    npm = sh(node.npm_bin)

    clone_cmd = ""
    if repo_url:
        clone_cmd = (
            f"if [ ! -f \"$PROJECT_DIR/requirements.txt\" ]; then "
            f"git clone --branch {sh(branch)} {sh(repo_url)} \"$PROJECT_DIR\"; "
            f"fi && "
        )

    frontend_setup = ""
    if node.frontend:
        frontend_setup = f"cd frontend && {npm} install && cd .. && "

    pip_bootstrap = (
        f"if ! {py} -m pip --version >/dev/null 2>&1; then "
        f"if command -v curl >/dev/null 2>&1; then "
        f"curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; "
        f"elif command -v wget >/dev/null 2>&1; then "
        f"wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py; "
        f"else echo 'Missing curl/wget required to bootstrap pip' >&2; exit 1; fi; "
        f"{py} /tmp/get-pip.py --user --break-system-packages; "
        f"fi"
    )

    return (
        f"PROJECT_DIR={project_expr} && "
        f"mkdir -p \"$PROJECT_DIR\" && "
        f"{clone_cmd}"
        f"cd \"$PROJECT_DIR\" && "
        f"{pip_bootstrap} && "
        f"if {py} -m venv .venv >/dev/null 2>&1; then "
        f". .venv/bin/activate && "
        f"python -m pip install --upgrade pip && "
        f"python -m pip install -r requirements.txt; "
        f"else "
        f"{py} -m pip install --user --break-system-packages --upgrade pip && "
        f"{py} -m pip install --user --break-system-packages -r requirements.txt; "
        f"fi && "
        f"{frontend_setup}"
        f"echo BOOTSTRAP_OK"
    )


def bootstrap_cluster(nodes: List[NodeConfig], passwords: Dict[str, str], args: argparse.Namespace) -> None:
    for node in nodes:
        print(f"\n[{node.name}] Bootstrapping {node.user}@{node.host}...")
        client = connect(node, passwords[node.name])
        try:
            bootstrap_cmd = build_bootstrap_command(node, args.repo_url, args.repo_branch)
            code, out, err = run_remote(client, bootstrap_cmd)
            if code != 0:
                raise RuntimeError(f"Bootstrap failed: {err.strip() or out.strip()}")
            print(f"[{node.name}] bootstrap complete")
        finally:
            client.close()


def start_cluster(nodes: List[NodeConfig], passwords: Dict[str, str], args: argparse.Namespace) -> None:
    backend_nodes = [n for n in nodes if n.backend]
    shard_count = len(backend_nodes)
    index_by_name = {node.name: idx for idx, node in enumerate(backend_nodes)}
    failures: List[str] = []

    for node in nodes:
        print(f"\n[{node.name}] Connecting to {node.user}@{node.host}...")
        client: Optional[paramiko.SSHClient] = None
        try:
            client = connect(node, passwords[node.name])
            code, out, err = run_remote(client, "mkdir -p $HOME/tmp")
            if code != 0:
                failures.append(f"{node.name}: unable to prepare host: {err.strip() or out.strip()}")
                continue

            if node.backend:
                shard_index = index_by_name[node.name]
                backend_cmd = build_backend_command(node, shard_index, shard_count, args)
                code, out, err = run_remote(client, backend_cmd)
                if code != 0:
                    failures.append(f"{node.name}: backend start failed: {err.strip() or out.strip()}")
                else:
                    print(f"[{node.name}] backend started shard {shard_index}/{shard_count - 1}")

            if node.frontend:
                frontend_cmd = build_frontend_command(node)
                code, out, err = run_remote(client, frontend_cmd)
                if code != 0:
                    failures.append(f"{node.name}: frontend start failed: {err.strip() or out.strip()}")
                else:
                    print(f"[{node.name}] frontend started on port {node.port}")
        finally:
            if client is not None:
                client.close()

    if failures:
        raise RuntimeError("; ".join(failures))


def stop_cluster(nodes: List[NodeConfig], passwords: Dict[str, str]) -> None:
    stop_backend = "pkill -f continuous_improve_pipeline.py || true"
    stop_frontend = "pkill -f vite || true"

    for node in nodes:
        print(f"\n[{node.name}] Stopping services...")
        client = connect(node, passwords[node.name])
        try:
            run_remote(client, stop_backend)
            if node.frontend:
                run_remote(client, stop_frontend)
            print(f"[{node.name}] stop signal sent")
        finally:
            client.close()


def fetch_node_logs(
    node: NodeConfig,
    password: str,
    log_lines: int,
) -> Dict[str, object]:
    client = connect(node, password)
    project_expr = remote_path_expr(node.project_dir)

    try:
        backend_proc_cmd = (
            f"PROJECT_DIR={project_expr} && cd \"$PROJECT_DIR\" && "
            "pgrep -u \"$USER\" -af \"[c]ontinuous_improve_pipeline.py\" || true"
        )
        backend_log_cmd = (
            f"PROJECT_DIR={project_expr} && cd \"$PROJECT_DIR\" && "
            f"tail -n {int(log_lines)} logs/backend_{node.name}.log 2>/dev/null || true"
        )
        backend_state_cmd = (
            f"PROJECT_DIR={project_expr} && cd \"$PROJECT_DIR\" && "
            f"if [ -f outputs/continuous_state_{node.name}.json ]; then "
            f"cat outputs/continuous_state_{node.name}.json; "
            "elif [ -f outputs/continuous_state.json ]; then "
            "cat outputs/continuous_state.json; "
            "else echo '{}'; fi"
        )

        _, backend_proc_out, _ = run_remote(client, backend_proc_cmd)
        _, backend_log_out, _ = run_remote(client, backend_log_cmd)
        _, backend_state_out, _ = run_remote(client, backend_state_cmd)

        backend_processes = [line for line in backend_proc_out.splitlines() if line.strip()]

        backend_state: Dict[str, object] = {}
        try:
            parsed_state = json.loads(backend_state_out.strip() or "{}")
            if isinstance(parsed_state, dict):
                backend_state = parsed_state
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
                "running": len(backend_processes) > 0,
                "processes": backend_processes,
                "logTail": backend_log_out.strip(),
                "state": {
                    "status": backend_state.get("status"),
                    "updated_at": backend_state.get("updated_at"),
                    "current_best_cv_auc": backend_state.get("current_best_cv_auc"),
                    "iterations_completed": backend_state.get("iterations_completed"),
                },
            },
        }

        if node.frontend:
            frontend_proc_pattern = f"[v]ite|[h]ttp.server {int(node.port)}"
            frontend_proc_cmd = (
                f"PROJECT_DIR={project_expr} && cd \"$PROJECT_DIR\" && "
                f"pgrep -u \"$USER\" -af \"{frontend_proc_pattern}\" || true"
            )
            frontend_log_cmd = (
                f"PROJECT_DIR={project_expr} && cd \"$PROJECT_DIR\" && "
                f"tail -n {int(log_lines)} logs/frontend_{node.name}.log 2>/dev/null || true"
            )
            _, frontend_proc_out, _ = run_remote(client, frontend_proc_cmd)
            _, frontend_log_out, _ = run_remote(client, frontend_log_cmd)

            frontend_processes = [line for line in frontend_proc_out.splitlines() if line.strip()]
            payload["frontend"] = {
                "running": len(frontend_processes) > 0,
                "processes": frontend_processes,
                "logTail": frontend_log_out.strip(),
                "url": f"http://{node.host}:{node.port}",
            }

        return payload
    finally:
        client.close()


def sync_logs_cluster(nodes: List[NodeConfig], passwords: Dict[str, str], args: argparse.Namespace) -> None:
    records: List[Dict[str, object]] = []

    for node in nodes:
        print(f"\n[{node.name}] Collecting logs...")
        record = fetch_node_logs(node, passwords[node.name], args.log_lines)
        records.append(record)
        print(f"[{node.name}] logs collected")

    result = {
        "generatedAt": datetime.now().isoformat(),
        "servers": records,
    }

    out_path = Path(args.server_logs_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote server log snapshot to: {out_path}")


def status_cluster(nodes: List[NodeConfig], passwords: Dict[str, str], log_lines: int) -> None:
    for node in nodes:
        print(f"\n[{node.name}] Status")
        record = fetch_node_logs(node, passwords[node.name], log_lines)

        backend = record.get("backend", {})
        print(f"backend running: {backend.get('running', False)}")
        for proc in backend.get("processes", []):
            print(f"  {proc}")

        if node.frontend:
            frontend = record.get("frontend", {})
            print(f"frontend running: {frontend.get('running', False)}")
            print(f"frontend url: {frontend.get('url', '-')}")
            for proc in frontend.get("processes", []):
                print(f"  {proc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SSH orchestrator for non-overlapping distributed continuous improvement workers."
    )
    parser.add_argument(
        "action",
        choices=["sync", "bootstrap", "start", "sync-logs", "stop", "status"],
        help="Action to run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ops/cluster_nodes.json"),
        help="Path to cluster node configuration JSON",
    )
    parser.add_argument(
        "--local-project-dir",
        type=Path,
        default=Path("."),
        help="Local project directory used by sync action",
    )
    parser.add_argument("--train-file", type=str, default="data/train.csv", help="Train file path on remote host")
    parser.add_argument("--test-file", type=str, default="data/test.csv", help="Test file path on remote host")
    parser.add_argument(
        "--leaderboard-scores-file",
        type=str,
        default="frontend/src/data/leaderboard_scores.json",
        help="Leaderboard file path on remote host",
    )
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--confirm-splits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--parallel-jobs", type=int, default=-1)
    parser.add_argument("--min-improvement", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repo-url",
        type=str,
        default=None,
        help="Optional git URL cloned when project_dir does not exist",
    )
    parser.add_argument(
        "--repo-branch",
        type=str,
        default="main",
        help="Branch used with --repo-url clone",
    )
    parser.add_argument(
        "--server-logs-file",
        type=Path,
        default=Path("frontend/src/data/server_logs.json"),
        help="Local JSON file to write server runtime logs",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=40,
        help="Number of log lines to fetch per service",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")

    nodes = load_nodes(args.config)
    passwords = resolve_passwords(nodes)

    if args.action == "sync":
        sync_cluster(nodes, passwords, args)
        return

    if args.action == "bootstrap":
        bootstrap_cluster(nodes, passwords, args)
        return

    if args.action == "start":
        start_cluster(nodes, passwords, args)
        return

    if args.action == "sync-logs":
        sync_logs_cluster(nodes, passwords, args)
        return

    if args.action == "stop":
        stop_cluster(nodes, passwords)
        return

    status_cluster(nodes, passwords, args.log_lines)


if __name__ == "__main__":
    main()
