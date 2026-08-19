import os
import subprocess
import sys
import time
import locale
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
NEUPPS_DIR = ROOT_DIR / "NPSR_Seawall"
DEFAULT_OUTPUT_NAME = "learned_sheet.ply"


class NeuPPSError(RuntimeError):
    pass


def _find_python_executable() -> str:
    candidates = []

    env_python = os.environ.get("NEUPPS_PYTHON")
    if env_python:
        candidates.append(env_python)

    env_name = os.environ.get("NEUPPS_CONDA_ENV", "NISR_Seawall")
    user_profile = Path.home()
    candidates.extend([
        str(user_profile / "anaconda3" / "envs" / env_name / "python.exe"),
        str(user_profile / "miniconda3" / "envs" / env_name / "python.exe"),
    ])

    candidates.append(sys.executable)

    venv_dir = NEUPPS_DIR / ".venv"
    if os.name == "nt":
        candidates.append(str(venv_dir / "Scripts" / "python.exe"))
    else:
        candidates.append(str(venv_dir / "bin" / "python"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return sys.executable


def _find_conda_executable() -> str | None:
    candidates = []

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(conda_exe)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix_path = Path(conda_prefix)
        candidates.extend([
            str(prefix_path / "Scripts" / "conda.exe"),
            str(prefix_path.parent / "condabin" / "conda.bat"),
            str(prefix_path.parent / "Scripts" / "conda.exe"),
        ])

    user_profile = Path.home()
    candidates.extend([
        str(user_profile / "miniconda3" / "Scripts" / "conda.exe"),
        str(user_profile / "anaconda3" / "Scripts" / "conda.exe"),
        str(user_profile / "miniconda3" / "condabin" / "conda.bat"),
        str(user_profile / "anaconda3" / "condabin" / "conda.bat"),
    ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return None


def _build_command(input_path: str, output_dir: Path) -> list[str]:
    base_args = [
        "main.py",
        "--multi_patch",
        "--pretrain_then_train",
        "--result_dir",
        str(output_dir),
        "--pretrain_epochs",
        "1000",
        "--epochs",
        "5000",
        "--n_patches",
        "16",
        "--d_features",
        "88",
        "--M_per_patch",
        "4096",
        "--W",
        "512",
        "--N",
        "5000000",
        "--mesh_res",
        "200",
        "--file",
        input_path,
        "--D",
        "6",
        "--L",
        "0",
        "--beta",
        "100",
        "--mu",
        "0.08",
        "--gamma",
        "0",
        "--lam",
        "0",
        "--lam2",
        "0",
        "--log_every",
        "200",
        "--pretrain_loss",
        "l1",
    ]

    env_name = os.environ.get("NEUPPS_CONDA_ENV", "NISR_Seawall")
    python_exe = _find_python_executable()
    if python_exe and Path(python_exe).exists() and Path(python_exe).resolve() != Path(sys.executable).resolve():
        return [python_exe, *base_args]

    conda_exe = _find_conda_executable()
    if conda_exe:
        return [conda_exe, "run", "-n", env_name, "python", *base_args]

    return [python_exe, *base_args]


def _build_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CONDA_NO_PLUGINS"] = "true"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if os.name == "nt":
        env.setdefault("CHCP", "65001")
        env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "utf-8")
    return env


def _ensure_input_file(point_cloud_path: str, work_dir: str) -> str:
    src = Path(point_cloud_path)
    if not src.exists():
        raise NeuPPSError(f"Selected point cloud file does not exist: {point_cloud_path}")

    suffix = src.suffix.lower()
    if suffix in {".ply", ".xyz", ".txt", ".csv", ".pts", ".npy"}:
        return str(src)

    raise NeuPPSError(
        "NeuPPS supports point cloud inputs in .ply, .xyz, .txt, .csv, .pts, or .npy format."
    )


def run_neupps(point_cloud_path: str, progress_cb=None, cancel_cb=None):
    progress_cb = progress_cb or (lambda value: None)
    cancel_cb = cancel_cb or (lambda: False)
    status_cb = None
    if callable(progress_cb) and hasattr(progress_cb, "__self__"):
        status_cb = getattr(progress_cb.__self__, "status_cb", None)

    def _status(message: str):
        if callable(status_cb):
            status_cb(message)

    if cancel_cb():
        raise NeuPPSError("NeuPPS run cancelled before start.")

    if not NEUPPS_DIR.exists():
        raise NeuPPSError(f"NeuPPS directory not found: {NEUPPS_DIR}")

    _status(f"NeuPPS: validating input file {point_cloud_path}")
    input_path = _ensure_input_file(point_cloud_path, str(NEUPPS_DIR))
    input_stem = Path(input_path).stem
    output_dir = NEUPPS_DIR / "logs" / f"app_neupps_{input_stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_log_path = output_dir / "app_runner.log"
    actual_output_dir = output_dir
    _status(f"NeuPPS: output directory ready at {output_dir}")
    _status(f"NeuPPS: runner log file {runner_log_path}")

    command = _build_command(input_path, output_dir)
    _status("NeuPPS: launching external reconstruction process")
    _status(f"NeuPPS command: {' '.join(command)}")

    progress_cb(5)
    current_progress = 5
    output_lines = []
    return_code = None

    def _set_progress(value: int):
        nonlocal current_progress
        current_progress = max(current_progress, int(value))
        progress_cb(current_progress)

    with runner_log_path.open("a", encoding="utf-8", errors="replace") as runner_log:
        runner_log.write(f"NeuPPS input: {input_path}\n")
        runner_log.write(f"NeuPPS output dir: {output_dir}\n")
        runner_log.write(f"NeuPPS command: {' '.join(command)}\n")
        runner_log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(NEUPPS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
            env=_build_subprocess_env(),
        )

        def _terminate_process_tree():
            if process.poll() is not None:
                return
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    process.terminate()
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        last_output_time = time.monotonic()
        last_heartbeat_time = last_output_time

        try:
            while True:
                if cancel_cb():
                    _status("NeuPPS: cancellation requested, terminating process tree")
                    runner_log.write("NeuPPS: cancellation requested\n")
                    runner_log.flush()
                    _terminate_process_tree()
                    raise NeuPPSError("NeuPPS run cancelled.")

                line = process.stdout.readline() if process.stdout is not None else ""
                if line:
                    last_output_time = time.monotonic()
                    stripped = line.rstrip()
                    output_lines.append(stripped)
                    runner_log.write(stripped + "\n")
                    runner_log.flush()
                    _status(f"NeuPPS output: {stripped}")
                    lower = line.lower()
                    if "output directory:" in lower:
                        _, _, reported_dir = stripped.partition(":")
                        candidate_dir = Path(reported_dir.strip())
                        if candidate_dir.exists():
                            actual_output_dir = candidate_dir
                            _status(f"NeuPPS: detected actual output directory {actual_output_dir}")
                    if "starting training" in lower:
                        _status("NeuPPS: training phase started")
                        _set_progress(15)
                    elif "initialization pretraining complete" in lower:
                        _status("NeuPPS: pretraining phase complete")
                        _set_progress(45)
                    elif "saving results to" in lower:
                        _status("NeuPPS: saving reconstruction outputs")
                        _set_progress(85)
                    elif "run complete" in lower:
                        _status("NeuPPS: external process reported completion")
                        _set_progress(95)
                elif process.poll() is not None:
                    break
                else:
                    now = time.monotonic()
                    if now - last_heartbeat_time >= 10:
                        last_heartbeat_time = now
                        if current_progress < 90:
                            _set_progress(min(current_progress + 1, 90))
                        idle_seconds = int(now - last_output_time)
                        heartbeat = (
                            f"NeuPPS: process still running, waiting for output ({idle_seconds}s idle)"
                        )
                        _status(heartbeat)
                        runner_log.write(heartbeat + "\n")
                        runner_log.flush()
                    time.sleep(0.1)

            return_code = process.wait()
            runner_log.write(f"NeuPPS: process exit code {return_code}\n")
            runner_log.flush()
            _status(f"NeuPPS: external process exited with code {return_code}")
        finally:
            if cancel_cb():
                _terminate_process_tree()
            if process.stdout is not None:
                process.stdout.close()

    if return_code != 0:
        details = "\n".join(output_lines[-40:]).strip()
        if not details:
            details = (
                f"No subprocess output was captured. Check the runner log: {runner_log_path}"
            )
        if "UnicodeEncodeError" in details and "conda" in details.lower():
            raise NeuPPSError(
                "NeuPPS could not start through `conda run` because conda crashed while writing non-ASCII output to the Windows terminal encoding. "
                "This is a conda console-encoding issue, not a NeuPPS training hang. "
                "Set `NEUPPS_PYTHON` to the Python executable inside the `NISR_Seawall` environment to bypass `conda run`, or launch the app from a UTF-8 terminal.\n"
                + details
            )
        if "No module named 'torch'" in details or 'No module named "torch"' in details:
            raise NeuPPSError(
                "NeuPPS could not find PyTorch in the runtime environment. "
                "Create or activate the `NISR_Seawall` conda environment from "
                "`NPSR_Seawall/environment.yaml`, or set `NEUPPS_CONDA_ENV` / `NEUPPS_PYTHON` "
                "to a Python environment that has NeuPPS dependencies installed.\n"
                + details
            )
        if "An unexpected error has occurred. Conda has prepared the above report." in details:
            raise NeuPPSError(
                "NeuPPS could not start through conda because conda itself failed before launching Python. "
                "The runner now disables conda plugins automatically, but your conda installation is still erroring. "
                "Try running the logged command manually, or set `NEUPPS_PYTHON` to the Python executable inside the `NISR_Seawall` environment to bypass `conda run`.\n"
                + details
            )
        raise NeuPPSError("NeuPPS failed.\n" + details)

    mesh_path = actual_output_dir / DEFAULT_OUTPUT_NAME
    if not mesh_path.exists():
        raise NeuPPSError(
            f"NeuPPS finished but did not produce {DEFAULT_OUTPUT_NAME} in {actual_output_dir}"
        )

    _status(f"NeuPPS: mesh generated at {mesh_path}")
    _set_progress(100)
    return {
        "mesh_path": str(mesh_path),
        "output_dir": str(actual_output_dir),
        "runner_log_path": str(runner_log_path),
        "log_tail": output_lines[-40:],
    }


def import_neupps_mesh(mesh_path: str):
    from io_utils.ply_io import load_file

    layer = load_file(mesh_path)
    return layer


__all__ = ["NeuPPSError", "run_neupps", "import_neupps_mesh"]
