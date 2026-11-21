#!/usr/bin/env python3
"""
Adversarial-training defense (proposal methodology step 5, stretch goal):
"Retrain the classifier on a mix of original and perturbed samples and measure
whether this basic adversarial-training defense recovers detection accuracy."

HOW IT WORKS
  1. Reconstruct the exact original train/val split (same seed, same loader, same
     --data/--min-attack-rows/--split as the baseline run, read automatically from
     artifacts/baseline/run_config.json so it can't silently drift from the real run).
  2. Take the training rows that are attacks, and perturb them with PGD using the
     SAME attacker-controllable feature budget as awid3_attack.py (imported directly,
     not re-implemented, so the two scripts can never disagree about the threat model).
  3. Augment the training set with those perturbed copies (labels unchanged -- a
     perturbed attack is still an attack) and train a FRESH Random Forest and MLP
     from scratch on original+perturbed data.
  4. Evaluate the defended models two ways against the untouched held-out test set:
       (a) on the clean test set -- did defending cost any clean accuracy?
       (b) on a FRESH adversarial version of the test set, attacked again from
           scratch against the DEFENDED model's own gradients (an adaptive
           re-attack -- evaluating a defense against the stale, pre-defense
           perturbations would overstate how well it works).

This script imports awid3_baseline.py and awid3_attack.py as modules (run from the
same folder) rather than duplicating their logic, so the data split and the
perturbation budget are guaranteed identical to the baseline and attack runs.

OUTPUT (into --out, default artifacts/defense/):
  defense_metrics.json  -- clean and adversarial accuracy/recall, baseline vs. defended,
                            for RF and MLP
  rf_defended.joblib, mlp_defended.pt, scaler is unchanged (reused from baseline)
"""
import argparse
import json
import os

import joblib
import numpy as np
import torch

import awid3_baseline as bl
import awid3_attack as atk

SEED = 42


def load_run_config(art_dir):
    with open(os.path.join(art_dir, "run_config.json")) as f:
        return json.load(f)


def reconstruct_train_val(cfg, label_col=None):
    """Re-run the baseline's own loader/feature-builder/splitter with the exact
    config the real baseline run used, so Xtr/ytr/Xva/yva match it row-for-row."""
    df, lc = bl.load_data(cfg["data"], cfg["max_rows_per_file"], cfg["max_files"],
                          label_col or cfg.get("label_col"),
                          cfg.get("per_group"), cfg.get("min_attack_rows", 0))
    X, y, attack_type, groups = bl.build_features(df, lc)
    tr, va, te = bl.split_data(X, y, groups, cfg.get("split", "blocks"))
    return X.iloc[tr], y[tr], attack_type[tr], X.iloc[va], y[va], list(X.columns)


def craft_training_perturbations(mlp, Xtr_s, ytr, features, scaler, sentinel, pgd_steps):
    """PGD-perturb the attack rows of the (already scaled) training set, using the
    same budget/realism logic as awid3_attack.py."""
    mask, scaled_bound, kind, rel_bound, budget_record, idx = atk.build_budget(features, scaler, sentinel)
    atk_idx = np.where(ytr == 1)[0]
    Xs = Xtr_s[atk_idx]
    X_raw = scaler.inverse_transform(Xs)
    active = atk.sentinel_active_mask(X_raw, features, sentinel, mask)
    row_mask = np.tile(mask, (Xs.shape[0], 1)) & active

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(np.ones(len(atk_idx)), dtype=torch.float32)
    mask_row_t = torch.tensor(row_mask, dtype=torch.float32)
    bound_t = torch.tensor(np.tile(scaled_bound, (Xs.shape[0], 1)), dtype=torch.float32) * mask_row_t
    bound_t = torch.where(mask_row_t.bool(), bound_t, torch.zeros_like(bound_t))
    for i in range(len(features)):
        if kind[i] == "binary" or rel_bound[i] > 0:
            col_active = torch.tensor(row_mask[:, i], dtype=torch.bool)
            bound_t[col_active, i] = float("inf")

    mlp.eval()
    Xadv_s = atk.pgd_attack(mlp, Xt, yt, mask_row_t, bound_t, steps=pgd_steps).numpy()
    X_raw_adv = scaler.inverse_transform(Xadv_s)
    X_raw_adv = atk.postprocess_realism(X_raw_adv, X_raw, features, idx, kind, rel_bound, active)
    return scaler.transform(X_raw_adv), budget_record


def adversarial_test_set(mlp, X_test_s, y_test, features, scaler, sentinel, pgd_steps):
    """Fresh adaptive PGD attack against whichever model is passed in (used to
    re-attack the DEFENDED model, not reuse the stale pre-defense perturbations)."""
    mask, scaled_bound, kind, rel_bound, budget_record, idx = atk.build_budget(features, scaler, sentinel)
    atk_idx = np.where(y_test == 1)[0]
    Xs = X_test_s[atk_idx]
    X_raw = scaler.inverse_transform(Xs)
    active = atk.sentinel_active_mask(X_raw, features, sentinel, mask)
    row_mask = np.tile(mask, (Xs.shape[0], 1)) & active

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(np.ones(len(atk_idx)), dtype=torch.float32)
    mask_row_t = torch.tensor(row_mask, dtype=torch.float32)
    bound_t = torch.tensor(np.tile(scaled_bound, (Xs.shape[0], 1)), dtype=torch.float32) * mask_row_t
    bound_t = torch.where(mask_row_t.bool(), bound_t, torch.zeros_like(bound_t))
    for i in range(len(features)):
        if kind[i] == "binary" or rel_bound[i] > 0:
            col_active = torch.tensor(row_mask[:, i], dtype=torch.bool)
            bound_t[col_active, i] = float("inf")

    mlp.eval()
    Xadv_s = atk.pgd_attack(mlp, Xt, yt, mask_row_t, bound_t, steps=pgd_steps).numpy()
    X_raw_adv = scaler.inverse_transform(Xadv_s)
    X_raw_adv = atk.postprocess_realism(X_raw_adv, X_raw, features, idx, kind, rel_bound, active)
    X_test_adv_s = X_test_s.copy()
    X_test_adv_s[atk_idx] = scaler.transform(X_raw_adv)
    return X_test_adv_s


def eval_models(rf, mlp, X_s, y, attack_type, device):
    with torch.no_grad():
        p_mlp = (torch.sigmoid(mlp(torch.tensor(X_s, dtype=torch.float32, device=device))) >= 0.5).cpu().numpy().astype(int)
    p_rf = rf.predict(X_s)
    return {"random_forest": atk.metrics_block(y, p_rf, attack_type),
            "mlp": atk.metrics_block(y, p_mlp, attack_type)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/baseline")
    ap.add_argument("--out", default="artifacts/defense")
    ap.add_argument("--data", nargs="+", default=None, help="override; default reads run_config.json")
    ap.add_argument("--pgd-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--rf-trees", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features, sentinel, scaler, rf, mlp_state, test = atk.load_baseline(a.artifacts)
    mlp = atk.build_mlp(len(features), mlp_state)
    X_test_s, y_test, attack_type_test = test["X_test"], test["y_test"], test["attack_type"]

    cfg = load_run_config(a.artifacts)
    if a.data:
        cfg["data"] = a.data
    print("Reconstructing the original train/val split (same seed as the baseline run)...")
    Xtr_df, ytr, at_tr, Xva_df, yva, features_check = reconstruct_train_val(cfg)
    assert features_check == features, (
        "Reconstructed feature columns don't match artifacts/baseline/features.json -- "
        "the data/ folder used for --data must be the SAME pieces the baseline run used.")
    sentinel_vec = np.array([sentinel.get(f, 0.0) for f in features])
    Xtr = Xtr_df.fillna(dict(zip(features, sentinel_vec))).to_numpy()
    Xva = Xva_df.fillna(dict(zip(features, sentinel_vec))).to_numpy()
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    print(f"Train rows: {len(ytr):,} ({int(ytr.sum()):,} attack).  "
          f"Crafting PGD perturbations against the ORIGINAL model for the attack rows...")
    Xtr_adv_s, budget_record = craft_training_perturbations(mlp, Xtr_s, ytr, features, scaler, sentinel, a.pgd_steps)
    atk_idx = np.where(ytr == 1)[0]
    Xtr_aug_s = np.vstack([Xtr_s, Xtr_adv_s])
    ytr_aug = np.concatenate([ytr, ytr[atk_idx]])
    print(f"Augmented training set: {len(ytr):,} original + {len(atk_idx):,} perturbed "
          f"= {len(ytr_aug):,} rows")

    print("Retraining Random Forest on original+perturbed data...")
    rf_def = bl.RandomForestClassifier(n_estimators=a.rf_trees, n_jobs=-1, random_state=SEED,
                                       class_weight="balanced_subsample").fit(Xtr_aug_s, ytr_aug)

    print("Retraining MLP on original+perturbed data...")
    mlp_def = bl.train_mlp(Xtr_aug_s, ytr_aug, Xva_s, yva, a.epochs, a.batch, a.lr, a.patience, device)

    print("Evaluating clean (unperturbed) test performance, baseline vs. defended...")
    results = {
        "n_train_original": int(len(ytr)), "n_train_augmented": int(len(ytr_aug)),
        "clean": {
            "baseline": eval_models(rf, atk.build_mlp(len(features), mlp_state), X_test_s, y_test, attack_type_test, device),
            "defended": eval_models(rf_def, mlp_def, X_test_s, y_test, attack_type_test, device),
        },
    }

    print("Crafting a FRESH adaptive PGD attack against the defended MLP (not reusing "
          "the pre-defense perturbations)...")
    X_test_adv_vs_defended = adversarial_test_set(mlp_def, X_test_s, y_test, features, scaler, sentinel, a.pgd_steps)
    X_test_adv_vs_baseline = adversarial_test_set(atk.build_mlp(len(features), mlp_state), X_test_s, y_test,
                                                   features, scaler, sentinel, a.pgd_steps)
    results["adversarial"] = {
        "baseline_under_its_own_attack": eval_models(rf, atk.build_mlp(len(features), mlp_state),
                                                      X_test_adv_vs_baseline, y_test, attack_type_test, device),
        "defended_under_fresh_attack": eval_models(rf_def, mlp_def, X_test_adv_vs_defended,
                                                    y_test, attack_type_test, device),
        "defended_under_original_stale_attack": eval_models(rf_def, mlp_def, X_test_adv_vs_baseline,
                                                             y_test, attack_type_test, device),
    }
    results["perturbation_budget"] = budget_record

    joblib.dump(rf_def, os.path.join(a.out, "rf_defended.joblib"))
    torch.save(mlp_def.state_dict(), os.path.join(a.out, "mlp_defended.pt"))
    json.dump(results, open(os.path.join(a.out, "defense_metrics.json"), "w"), indent=1)

    def r(d, k): return d[k]["recall"] if d[k]["recall"] is not None else float("nan")
    print(f"\n{'':28s} {'RF recall':>10s} {'MLP recall':>11s}")
    print(f"{'clean / baseline model':28s} {r(results['clean']['baseline'],'random_forest'):10.4f} "
          f"{r(results['clean']['baseline'],'mlp'):11.4f}")
    print(f"{'clean / defended model':28s} {r(results['clean']['defended'],'random_forest'):10.4f} "
          f"{r(results['clean']['defended'],'mlp'):11.4f}")
    print(f"{'attacked / baseline model':28s} "
          f"{r(results['adversarial']['baseline_under_its_own_attack'],'random_forest'):10.4f} "
          f"{r(results['adversarial']['baseline_under_its_own_attack'],'mlp'):11.4f}")
    print(f"{'attacked / defended model':28s} "
          f"{r(results['adversarial']['defended_under_fresh_attack'],'random_forest'):10.4f} "
          f"{r(results['adversarial']['defended_under_fresh_attack'],'mlp'):11.4f}")
    print(f"\nSaved results to {a.out}/")


if __name__ == "__main__":
    main()
