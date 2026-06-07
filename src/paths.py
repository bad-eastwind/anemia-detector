"""Central paths with environment-variable overrides (for Kaggle / HPC portability).

Defaults reproduce the original repo layout exactly (no behavior change). Override on Kaggle:
  ANEMIA_DATA_ROOT=/kaggle/input/<dataset>/dataset anemia
  ANEMIA_OUT_ROOT=/kaggle/working/outputs
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get("ANEMIA_DATA_ROOT",
                           os.path.abspath(os.path.join(_HERE, "..", "dataset anemia")))
OUT_ROOT = os.environ.get("ANEMIA_OUT_ROOT",
                          os.path.abspath(os.path.join(_HERE, "..", "outputs")))

CACHE = os.path.join(OUT_ROOT, "cache")
MANIFEST = os.path.join(OUT_ROOT, "manifest.csv")
FEATURES = os.path.join(OUT_ROOT, "anemia_features.csv")
SEG_OUT = os.path.join(OUT_ROOT, "seg")
CLF_OUT = os.path.join(OUT_ROOT, "clf")
SSL_OUT = os.path.join(OUT_ROOT, "ssl")
