#!/usr/bin/env python3
"""
scripts/port_forward.py

Purpose
- Start/stop/status Kubernetes port-forwards in an automated way for local demo/screenshots.
- Uses PID files to manage background kubectl port-forward processes.

Usage examples
- Start all:
    python scripts/port_forward.py start
- Stop all:
    python scripts/port_forward.py stop
- Status:
    python scripts/port_forward.py status
- Start one:
    python scripts/port_forward.py start mlflow
"""

from __future__ import annotations

import os
import sys
import signal
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_DIR = os.path.join(BASE_DIR, ".pids")
os.makedirs(PID_DIR, exist_ok=True)


@dataclass(frozen=True)
class PortForwardSpec:
    name: str
    namespace: str
    resource: str  # service/<name> or pod/<name>
    local_port: int
    remote_port: int


# Update these to match your services/namespaces
SPECS: Dict[str, PortForwardSpec] = {
    "catsdogs-api": PortForwardSpec(
        name="catsdogs-api",
        namespace="default",
        resource="service/catsdogs-api",
        local_port=8000,
        remote_port=8000,
    ),
    "mlflow": PortForwardSpec(
        name="mlflow",
        namespace="default",
        resource="service/mlflow",
        local_port=5000,
        remote_port=5000,
    ),
    "minio-api": PortForwardSpec(
        name="minio-api",
        namespace="default",
        resource="service/minio",
        local_port=9000,
        remote_port=9000,
    ),
    "minio-console": PortForwardSpec(
        name="minio-console",
        namespace="default",
        resource="service/minio",
        local_port=9001,
        remote_port=9001,
    ),
    "prometheus": PortForwardSpec(
        name="prometheus",
        namespace="monitoring",
        resource="service/prometheus",
        local_port=9090,
        remote_port=9090,
    ),
    "grafana": PortForwardSpec(
        name="grafana",
        namespace="monitoring",
        resource="service/grafana",
        local_port=3000,
        remote_port=3000,
    ),
}


def pid_file(name: str) -> str:
    return os.path.join(PID_DIR, f"{name}.pid")


def is_pid_running(pid: int) -> bool:
    """
    Check whether a PID exists.
    Works on Unix-like systems. For Windows, this method is not reliable.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but we may not have permission to signal it
        return True
    except OSError:
        return False


def read_pid(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return int(raw)
    except FileNotFoundError:
        return None
    except ValueError:
        return None
    except OSError:
        return None


def write_pid(path: str, pid: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(pid))


def remove_pidfile_safely(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError:
        return


def start_one(spec: PortForwardSpec) -> None:
    path = pid_file(spec.name)
    existing_pid = read_pid(path)

    if existing_pid is not None and is_pid_running(existing_pid):
        print(f"[SKIP] {spec.name} already running (PID {existing_pid})")
        return

    # If pidfile exists but process is dead, remove stale file.
    if existing_pid is not None and not is_pid_running(existing_pid):
        remove_pidfile_safely(path)

    cmd = [
        "kubectl",
        "-n",
        spec.namespace,
        "port-forward",
        spec.resource,
        f"{spec.local_port}:{spec.remote_port}",
    ]

    # Start in background with stdout/stderr redirected (prevents terminal lock)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # creates separate process group
    )

    write_pid(path, proc.pid)
    print(
        f"[STARTED] {spec.name} -> http://localhost:{spec.local_port} "
        f"(PID {proc.pid}, {spec.namespace}/{spec.resource} {spec.local_port}:{spec.remote_port})"
    )


def stop_one(spec: PortForwardSpec) -> None:
    path = pid_file(spec.name)
    pid = read_pid(path)

    if pid is None:
        print(f"[SKIP] {spec.name} not running (no pidfile)")
        return

    if not is_pid_running(pid):
        print(f"[CLEANUP] {spec.name} pidfile exists but PID {pid} not running")
        remove_pidfile_safely(path)
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"[STOPPED] {spec.name} (PID {pid})")
    except ProcessLookupError:
        print(f"[CLEANUP] {spec.name} PID {pid} not found")
    except PermissionError:
        print(f"[FAILED] {spec.name} (PID {pid}) permission denied")
    except OSError as e:
        print(f"[FAILED] {spec.name} (PID {pid}) OS error: {e}")
    finally:
        remove_pidfile_safely(path)


def status_one(spec: PortForwardSpec) -> None:
    path = pid_file(spec.name)
    pid = read_pid(path)

    if pid is None:
        print(f"[DOWN] {spec.name} (no pidfile)")
        return

    if is_pid_running(pid):
        print(f"[UP]   {spec.name} (PID {pid}) -> http://localhost:{spec.local_port}")
    else:
        print(f"[DOWN] {spec.name} (stale pidfile PID {pid})")
        remove_pidfile_safely(path)


def usage() -> None:
    names = ", ".join(sorted(SPECS.keys()))
    print(
        "Usage:\n"
        "  python scripts/port_forward.py start [name|all]\n"
        "  python scripts/port_forward.py stop [name|all]\n"
        "  python scripts/port_forward.py status [name|all]\n\n"
        f"Names: {names}"
    )


def main() -> int:
    if len(sys.argv) < 2:
        usage()
        return 2

    action = sys.argv[1].lower()
    target = sys.argv[2].lower() if len(sys.argv) >= 3 else "all"

    if target != "all" and target not in SPECS:
        print(f"[ERROR] Unknown target '{target}'")
        usage()
        return 2

    specs = list(SPECS.values()) if target == "all" else [SPECS[target]]

    if action == "start":
        for s in specs:
            start_one(s)
        return 0

    if action == "stop":
        for s in specs:
            stop_one(s)
        return 0

    if action == "status":
        for s in specs:
            status_one(s)
        return 0

    print(f"[ERROR] Unknown action '{action}'")
    usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
