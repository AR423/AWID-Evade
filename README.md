# Adversarial Evasion Against ML-Based Wi-Fi IDS (AWID3)

CIT 53200 group project. We train ML detectors (Random Forest, small neural net) on AWID3-CSV
Deauth and Evil_Twin traffic, then test how much constrained feature perturbations reduce detection.

## Setup
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Data (not stored in this repo)
Request AWID3 (CSV) from https://icsdweb.aegean.gr/awid/download and download a few 50,000-row
pieces (e.g. `Deauth_22.csv`, `Evil_Twin_40.csv`) into a local `data/` folder.

## Run
```
python inspect_awid3.py data/Deauth_22.csv       # look at columns and labels
python awid3_baseline.py --data data/ --out artifacts/
```

## Files
- `awid3_common.py`   shared loading/cleaning helpers
- `inspect_awid3.py`  column and label report for one CSV
- `awid3_baseline.py` Random Forest + MLP baseline, per-attack metrics
- `artifacts/`        results (`metrics.json`, `feature_importance_rf.csv`, `features.json`)

## Team workflow
Work on a branch (`git checkout -b yourname/feature`), commit small changes, open a pull request,
and run `git pull` before you start work.
