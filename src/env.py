import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_required(required_keys, env_path=None):
    env_path = Path(env_path) if env_path is not None else REPO_ROOT / ".env"
    if not env_path.exists():
        raise RuntimeError(
            f"Missing required .env file at {env_path}. "
            "Copy .env.example to .env and edit it for this machine."
        )

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise RuntimeError(f"Invalid .env line: {raw_line!r}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ[key] = value

    missing = [key for key in required_keys if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing required .env variable(s): {', '.join(missing)}")

    return {key: os.environ[key] for key in required_keys}
