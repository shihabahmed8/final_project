import random, shutil
from pathlib import Path

random.seed(42)
SRC = Path("data/train")
VAL = Path("data/val")
VAL_RATIO = 0.2

def main():
    classes = [d.name for d in SRC.iterdir() if d.is_dir()]
    print("Classes:", classes)
    for cname in classes:
        src_dir = SRC / cname
        val_dir = VAL / cname
        val_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in src_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        random.shuffle(files)
        n_val = int(len(files) * VAL_RATIO)
        for p in files[:n_val]:
            shutil.copy(str(p), str(val_dir / p.name))
        print(f"{cname}: copied {n_val} to val")

if __name__ == "__main__":
    main()
