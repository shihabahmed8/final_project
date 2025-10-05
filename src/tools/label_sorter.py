from pathlib import Path
from PIL import Image
import shutil

UNLABELED_DIR = Path("data/unlabeled")

CLASSES = [
    ("Unripe", "1"),
    ("Ripe", "2"),
    ("Overripe", "3"),
    ("BruisedDamaged", "4"),
    ("HealthyLeaf", "5"),
]

DEST_ROOT = Path("data/train")
DEST_ROOT.mkdir(parents=True, exist_ok=True)
for cname, _ in CLASSES:
    (DEST_ROOT / cname).mkdir(parents=True, exist_ok=True)

def ask_label(img_path):
    img = Image.open(img_path).convert("RGB")
    img.show()
    print("\nFile:", img_path)
    for cname, key in CLASSES:
        print(f"[{key}] {cname}")
    key = input("Class? (or 's' skip): ").strip().lower()
    img.close()
    return key

def main():
    files = [p for p in UNLABELED_DIR.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    print("Found", len(files), "images.")
    for p in files:
        key = ask_label(p)
        if key == "s":
            continue
        match = [c for c,k in CLASSES if k == key]
        if not match:
            print("Invalid key, skipping.")
            continue
        cname = match[0]
        dst = DEST_ROOT / cname / p.name
        shutil.move(str(p), str(dst))
        print("Moved ->", dst)

if __name__ == "__main__":
    main()
