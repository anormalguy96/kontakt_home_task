
import sys
import os
import pathlib

print("Current Working Directory:", os.getcwd())
print("Sys Path:")
for p in sys.path:
    print(f"  {p}")

print("\nFile resolution:")
try:
    from kontakt_qc import types
    print(f"kontakt_qc.types file: {types.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")

print("\nDirectory Listing of github_repos:")
try:
    # Go up to github_repos
    # assuming we are in task1/src/kontakt_qc
    # or running from root
    # let's try to find github_repos
    path = pathlib.Path(os.getcwd())
    while path.name != "github_repos" and path.parent != path:
        path = path.parent
    
    if path.name == "github_repos":
        print(f"Found github_repos at: {path}")
        print("Contents:")
        for item in path.iterdir():
            print(f"  {item.name}")
    else:
        print("Could not find github_repos in ancestry of " + os.getcwd())
except Exception as e:
    print(f"Error checking directories: {e}")
