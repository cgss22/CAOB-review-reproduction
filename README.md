# CAOB review reproduction

Place the released data directory alongside src. From the repository root, use Python 3.12 and install the pinned requirements.

```sh
python -m pip install -r requirements.txt
python src/CAOB_Review_Full.py --check
python src/CAOB_Review_Full.py
```

The runner creates a timestamped directory under result. Existing inputs include archived evidence layers, predictions, and model weights. The workflow starts from those prepared layers and does not regenerate the Surfer interpolation.

The release inventory identifies input files and outputs from the completed integrated run on 2026-11-21. Those outputs predate the reorganized English output layout of CAOB_Review_Full.py. The two November 23 runs were still marked running when the inventory was generated.

Data and result archive download links will be added after publication.
