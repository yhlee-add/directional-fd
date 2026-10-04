import os
import sys


def _venv_python(env_var: str, default: str) -> str:
    """
    Resolve the Python of an isolated venv, in order: $<env_var>, the Dockerfile
    default venv, then the current interpreter.
    """
    venv = os.environ.get(env_var)
    if venv is None and os.path.isdir(default):
        venv = default
    return os.path.join(venv, ".venv", "bin", "python") if venv else sys.executable


def laproteina_python() -> str:
    """
    Single source of truth for the Python that has La-Proteina's deps.
    """
    return _venv_python("LAPROTEINA_VENV", "/ddiff-base/py3.10-torch2.7.0")
