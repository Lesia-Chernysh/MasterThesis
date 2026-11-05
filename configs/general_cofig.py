from pathlib import Path

# Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'

# Semantic lens
DATASET_NAME = "ifigotin/imagenetmini-1000" # "ifigotin/imagenetmini-1000" or "titericz/imagenet1k-val"
EMBEDDING_MODEL = "SigLipV2"
K_ACT_NEURONS = 500  # how many most activating neurons will be visualized
N_SAMPLES = 10
THRESHOLD_ACT = 0.5 # mask and crop based on this threshold