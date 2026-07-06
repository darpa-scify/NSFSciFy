import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


def add_src_to_path() -> None:
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def resolve_root(root_dir: str | os.PathLike[str] | None = None) -> Path:
    if root_dir is None:
        return REPO_ROOT
    path = Path(root_dir).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_under_root(root_dir: Path, path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return root_dir / resolved


def load_repo_env() -> None:
    if not ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        with ENV_FILE.open("r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
        return
    load_dotenv(ENV_FILE)


def get_default_root() -> str:
    return os.getenv("NSFSCIFY_ROOT_DIR", str(REPO_ROOT))


def get_default_wandb_project() -> str:
    return os.getenv("WANDB_PROJECT", "nsf")


def configure_hf_home(root_dir: Path) -> None:
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = str(resolve_under_root(root_dir, hf_home))
    else:
        os.environ["HF_HOME"] = str(root_dir / ".cache" / "huggingface")


def resolve_model_path(model_name: str, root_dir: Path) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_absolute():
        return str(model_path)
    if model_path.parts and model_path.parts[0] in {".", "..", "models", "checkpoints"}:
        return str(root_dir / model_path)
    return model_name
