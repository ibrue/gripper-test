"""Entry point for the standalone umi.app bundle.

Launches the studio with a sensible default port (auto-detected) and the
default dataset folder (~/umi-data). The CLI subcommands in gripper.py
still work from the terminal — this script is only used when launched
from the .app icon.
"""

from gripper import guess_default_port
from studio import run_studio


def main() -> int:
    return run_studio(default_port=guess_default_port(), dataset_root="~/umi-data")


if __name__ == "__main__":
    raise SystemExit(main())
