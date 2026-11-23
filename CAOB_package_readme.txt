CAOB compact reproduction package

Keep all .partNNNN files, parts_manifest.json, restore_caob_25mib.py,
CAOB_Review_Full.py, and requirements.txt together. Do not rename the parts.

Run:
  python restore_caob_25mib.py
  python -m pip install -r requirements.txt
  python CAOB_Review_Full.py --check
  python CAOB_Review_Full.py

The restore script verifies each part and the reassembled ZIP against SHA-256.
The script can recreate omitted intermediate results from the included data.
See omitted_result_files.csv for exclusions. Archived training graphs are
distributed as JPEGs. Archived prediction tables used by ROC and P-A retain
the original coordinate, label and probability columns.
