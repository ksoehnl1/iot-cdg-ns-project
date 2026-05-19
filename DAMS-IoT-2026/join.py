import sys
import glob

# claude made this utility file, verified it works by checking the
# two hash values of the pre-split and post join file.

if len(sys.argv) != 2:
    print("Usage: python join.py <file.csv>")
    sys.exit(1)

base = sys.argv[1]
parts = sorted(glob.glob(f"{base}.part*"))

if not parts:
    print(f"No parts found for {base}")
    sys.exit(1)

with open(base, "wb") as outfile:
    for part in parts:
        with open(part, "rb") as infile:
            outfile.write(infile.read())
        print(f"  {part}")

print(f"Successfully joined {len(parts)} parts into {base}!")
