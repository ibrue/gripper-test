"""py2app build config for the umi standalone macOS app.

Usage (from a clean venv):
    pip install -r requirements.txt -r requirements-build.txt
    python setup_app.py py2app
The bundle ends up in dist/umi.app — drag it into /Applications.

Notes / gotchas (Apple Silicon):
    * Tk needs to be present at build time. ``brew install python-tk@3.14``
      gives you a usable interpreter; build with that python explicitly,
      not the macOS system python.
    * If you see "ModuleNotFoundError: No module named 'PIL._tkinter_finder'"
      at launch, ``pip install --upgrade pyobjc Pillow``.
    * For a notarised distributable, codesign + notarytool the .app after
      build; for personal use, right-click → Open the first time to
      bypass Gatekeeper.
"""

from setuptools import setup

APP = ["umi_main.py"]
DATA_FILES = []

OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "umi",
        "CFBundleDisplayName": "umi",
        "CFBundleIdentifier": "ai.umi.gripper-studio",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "NSHumanReadableCopyright": "© 2026 umi",
        # macOS asks for these the first time the app uses each capability.
        "NSCameraUsageDescription":
            "umi reads the Insta360 camera feed for live monocular SLAM.",
        "NSBluetoothAlwaysUsageDescription":
            "umi talks to the Insta360 X3 over Bluetooth to start/stop "
            "on-card recording and read battery.",
        "NSBluetoothPeripheralUsageDescription":
            "umi talks to the Insta360 X3 over Bluetooth.",
        "NSMicrophoneUsageDescription":
            "Some camera capture paths request the microphone alongside video.",
        # Don't require Retina-only rendering; matplotlib needs both.
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
    "packages": [
        "tkinter", "matplotlib", "numpy", "cv2", "PIL",
        "dynamixel_sdk",
    ],
    "includes": [
        "studio", "slam", "imu", "dataset", "x3_control",
    ],
    # bleak / telemetry-parser-py are optional; if installed at build time
    # they'll be picked up automatically.
    "optional_packages": ["bleak", "telemetry_parser"],
    # Launch directly into the studio.
    "extra_scripts": [],
}

setup(
    name="umi",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
    py_modules=["gripper", "studio", "slam", "imu", "dataset", "x3_control", "umi_main"],
)
