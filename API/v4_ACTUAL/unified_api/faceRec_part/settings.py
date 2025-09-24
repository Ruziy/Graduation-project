from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "known_faces"
STATIC_DIR = BASE_DIR / "static"

# Recognition parameters (tune if needed)
TOLERANCE = 0.50          # 0.68–0.80 is typical; 0.50 is strict
DETECT_SCALE = 0.25       # downscale for detector
UPSAMPLE = 1              # 0..2
DETECTOR = "hog"          # "hog" CPU; "cnn" only if dlib CUDA is available
ENC_MODEL_LIVE = "small"  # "small"/"large"
JITTER_LIVE = 0
MIN_FACE = 40             # min face size in full-res pixels

# WebSocket sending/receiving
SEND_FPS = 10            # how often client sends frames (client-side throttle too)
