# Adversarial Evasion Against ML-Based Wi-Fi IDS (AWID3)

CIT 53200 (Wireless Security) group project, Purdue University Indianapolis.

We train ML-based wireless intrusion detection (a Random Forest and a small MLP) on the AWID3
dataset's Deauth and Evil_Twin traffic, then measure how much small, constrained, physically
realistic perturbations to attack traffic reduce detection -- first as a numerical simulation
(feature-space), then as real injected and recaptured 802.11 frames (problem-space), and finally
whether adversarial training recovers the lost detection.

## Project stages and results

| Stage | Script(s) | Key result |
|---|---|---|
| 1. Baseline classifiers | `scripts/awid3_baseline.py` | RF: 99.98% accuracy / 99.75% recall. MLP: 99.89% accuracy / 99.97% recall. Trained on 2,955,156 rows, 48 features. |
| 2. Feature-space attack (FGSM/PGD) | `scripts/awid3_attack.py` | RF recall 99.75% -> 93.46% (FGSM) / 93.37% (PGD). MLP recall 99.97% -> 95.05% (FGSM) / 95.07% (PGD). |
| 3. Adversarial-training defense | `scripts/awid3_defense.py` | Clean recall barely moved (RF 99.75%->99.76%, MLP 99.97%->99.95%). Recall under a **fresh** adaptive attack recovered to RF 99.69%, MLP 96.33%. |
| 4. Problem-space attack (real Deauth injection) | `scripts/awid3_inject.py`, `scripts/awid3_extract.py` | MLP recall 100%->98% on real, physically injected and recaptured frames -- confirms the feature-space result transfers to the real world. RF showed 0% recall on *both* clean and perturbed captures; this is a documented hardware limitation (see below), not a genuine evasion. |

Full numbers for every stage are in `artifacts/<stage>/*.json`.

## Repository layout

```
scripts/      all code, run from the repo root (see Run, below)
  awid3_common.py     shared loading/cleaning helpers
  awid3_baseline.py   stage 1: Random Forest + MLP baseline
  awid3_attack.py     stage 2: FGSM/PGD feature-space attack
  awid3_defense.py    stage 3: adversarial-training defense
  awid3_inject.py     stage 4: Scapy Deauth frame injection (Kali + Alfa adapter, monitor mode)
  awid3_extract.py    stage 4: tshark feature extraction + scoring of a recaptured pcap
  scan_pieces.py      utility: scan AWID3 CSV pieces for attack-row counts
  inspect_awid3.py    utility: column/label report for one AWID3 CSV

artifacts/
  baseline/      stage 1 outputs: metrics.json, features.json, sentinel.json, feature_importance_rf.csv
                 (scaler.joblib/rf.joblib/mlp.pt are produced here too but are NOT committed -- see .gitignore)
  attack/        stage 2 outputs: perturbation_budget.json, attack_metrics.json, feature_shift_rf.csv
  defense/       stage 3 outputs: defense_metrics.json
  problem_space/ stage 4 outputs: clean.pcap, perturbed.pcap, problem_space_metrics.json
```

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

All commands below are run from the **repo root** (not from inside `scripts/`), so that the
scripts' default artifact paths resolve correctly:

```
python scripts/inspect_awid3.py data/Deauth_22.csv              # look at columns and labels
python scripts/awid3_baseline.py --data data/ --out artifacts/baseline

python scripts/awid3_attack.py  --artifacts artifacts/baseline --out artifacts/attack
python scripts/awid3_defense.py --artifacts artifacts/baseline --out artifacts/defense
```

The problem-space stage (4) needs a Linux machine with an external monitor-mode-capable Wi-Fi
adapter (we used an Alfa AWUS036ACM on Kali Linux) and root:

```
sudo airmon-ng check kill
sudo airmon-ng start wlan0        # creates wlan0mon
sudo python3 scripts/awid3_inject.py --iface wlan0mon --bssid <AP MAC> \
    --mode clean     --out artifacts/problem_space/clean.pcap
sudo python3 scripts/awid3_inject.py --iface wlan0mon --bssid <AP MAC> \
    --mode perturbed --out artifacts/problem_space/perturbed.pcap

python3 scripts/awid3_extract.py --pcap artifacts/problem_space/clean.pcap     --label 1 --artifacts artifacts/baseline
python3 scripts/awid3_extract.py --pcap artifacts/problem_space/perturbed.pcap --label 1 --artifacts artifacts/baseline
```

## Known limitation: problem-space Random Forest result

The real-hardware run above showed 0% RF recall on *both* the clean and perturbed capture. This
is not a genuine evasion result -- it's because injecting and capturing happened on the same
radio, so the injected frames never got a real over-the-air signal-strength reading (self-
transmitted frames aren't "received" the way someone else's traffic is). Those frames got filled
with an out-of-distribution sentinel value on RF's two highest-importance features (signal
strength, ~22% of its decision weight combined), which collapses RF's hard, axis-aligned splits
regardless of perturbation. MLP's smoother decision boundary was robust to this and still showed
a believable 100%->98% drop. See `artifacts/problem_space/problem_space_metrics.json` for the full
writeup, including the alternative capture setups (router, teammate's laptop) that were
investigated and ruled out.

## Team workflow

Work on a branch (`git checkout -b yourname/feature`), commit small changes, open a pull request,
and run `git pull` before you start work.
