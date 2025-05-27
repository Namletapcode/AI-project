import os
import sys

if len(sys.argv) != 3:
    print("Usage: python rename_files.py <search_string> <replace_string>")
    sys.exit(1)

search_str = sys.argv[1]
replace_str = sys.argv[2]

renamed = 0
for filename in os.listdir('.'):
    if os.path.isfile(filename) and search_str in filename:
        new_name = filename.replace(search_str, replace_str)
        os.rename(filename, new_name)
        print(f"Renamed: {filename} -> {new_name}")
        renamed += 1

if renamed == 0:
    print("No files matched.")
else:
    print(f"Renamed {renamed} file(s).")
