import kagglehub
import shutil
from pathlib import Path
path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
print("downloaded to:", path)

dest = Path("data/raw/cicids2017")
dest.mkdir(parents=True, exist_ok=True)
for f in Path(path).glob("*.csv"):
    shutil.copy(f, dest / f.name)
    print ("copied", f.name)
