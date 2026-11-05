import kagglehub 
import shutil
from pathlib import Path 

path = kagglehub.dataset_download("dhoogla/unswnb15")
print("downloaded to:", path) 

dest = Path("data/raw/unsw_nb15")
dest.mkdir(parents=True, exist_ok=True)
for f in Path(path) .rglob("*"):
    if f.is_file() and f.suffix in {".parquet", ".csv"}:
        shutil.copy(f, dest / f.name)
        print("Copied", f.name)