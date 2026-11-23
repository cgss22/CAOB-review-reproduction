"""Verify downloaded parts, then extract data and results beside this script."""
import hashlib
import json
import zipfile
from pathlib import Path

folder = Path(__file__).resolve().parent
manifest = json.loads((folder / "parts_manifest.json").read_text(encoding="utf8"))
for entry in manifest.values():
    temporary = folder / (entry["archive"] + ".partial")
    combined_hash = hashlib.sha256()
    with temporary.open("wb") as output:
        for part in entry["parts"]:
            block = (folder / part["name"]).read_bytes()
            if len(block) != part["bytes"] or hashlib.sha256(block).hexdigest() != part["sha256"]:
                raise ValueError(f"Incorrect or incomplete file: {part['name']}")
            combined_hash.update(block)
            output.write(block)
    if combined_hash.hexdigest() != entry["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {entry['archive']}")
    archive = folder / entry["archive"]
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as contents:
        if len(contents.infolist()) != entry["file_count"]:
            raise ValueError(f"File count mismatch: {archive.name}")
        for member in contents.namelist():
            if not (folder / member).resolve().is_relative_to(folder):
                raise ValueError(f"Unsafe archive path: {member}")
        contents.extractall(folder)
    print("Verified and extracted", archive.name)
