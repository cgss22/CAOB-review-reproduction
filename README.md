# CAOB review reproduction

Use Python 3.12 and install the pinned dependencies.

```sh
python -m pip install -r requirements.txt
python src/CAOB_Review_Full.py --check
python src/CAOB_Review_Full.py
```

Place the data directory alongside src. The runner creates a timestamped directory under result. It starts from the supplied evidence layers, saved predictions and model weights.

The release inventory lists the supplied inputs and results from CAOB_Review_Full_20261123_100459_268440. The original-feature RF map was rendered using the predictions from that run. The original-feature 1-D CNN map was added by running the saved model on the full grid and checking scores against the labelled predictions from that run.

See the Releases tab for the data and result archives and their verification instructions.
