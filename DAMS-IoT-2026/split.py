import sys
import os

# claude made this utility file

CHUNK_SIZE = 20 * 1024 * 1024

if len(sys.argv) != 2:
    print("Usage: python split.py <file.csv>")
    sys.exit(1)

input_file = sys.argv[1]

if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    sys.exit(1)

with open(input_file, "rb") as f:
    chunk_num = 0
    while True:
        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            break
        out_name = f"{input_file}.part{chunk_num:03}"
        with open(out_name, "wb") as out:
            out.write(chunk)
        size_mb = os.path.getsize(out_name) / (1024 * 1024)
        print(f"  {out_name} ({size_mb:.1f}MB)")
        chunk_num += 1

print(f"Successfully split {input_file} into {chunk_num} chunks!")
