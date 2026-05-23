import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from semantic_memory.training.colab import build_colab_training_script

if __name__ == "__main__":
    print(build_colab_training_script())
