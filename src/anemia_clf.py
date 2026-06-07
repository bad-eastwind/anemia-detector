"""Anemia classification from conjunctiva pallor features.

Two routes, both evaluated with pooled 5-fold CV AND leave-one-cohort-out (LOCO):
  (A) direct binary classification (anemic vs not)
  (B) Hgb regression -> WHO sex threshold (M<13, F<12) -> anemia label

Metrics: AUC, sensitivity, specificity, accuracy -- reported overall, per cohort, and the
sens/spec split already separates anemic vs non-anemic performance.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import FEATURES as FEAT, CLF_OUT as OUT
os.makedirs(OUT, exist_ok=True)

DROP = ["uid", "cohort", "anemia", "hgb", "sex"]


def metrics(y, proba, thr=0.5):
    y = np.asarray(y); pred = (np.asarray(proba) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(y)
    try:
        auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {"auc": auc, "sens": sens, "spec": spec, "acc": acc,
            "n": len(y), "n_pos": int(y.sum()), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def report(y, proba, cohort, thr=0.5):
    out = {"overall": metrics(y, proba, thr)}
    for c in ("India", "Italy"):
        mask = np.asarray(cohort) == c
        if mask.sum():
            out[c] = metrics(np.asarray(y)[mask], np.asarray(proba)[mask], thr)
    return out


def make_clf(name):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                              max_iter=400, l2_regularization=1.0,
                                              random_state=42)
    if name == "rf":
        return RandomForestClassifier(n_estimators=600, max_depth=None,
                                      min_samples_leaf=3, random_state=42, n_jobs=-1)
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
    raise ValueError(name)


def make_reg(name):
    if name == "hgb":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                             max_iter=400, l2_regularization=1.0,
                                             random_state=42)
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def who_label(hgb, sex_M):
    cut = np.where(np.asarray(sex_M) == 1, 13.0, 12.0)
    return (np.asarray(hgb) < cut).astype(int)


# --------------------------- protocols ---------------------------
def cv_classify(df, X, model, k=5):
    y = df["anemia"].values
    strat = (df["cohort"] + "_" + df["anemia"].astype(str)).values
    oof = np.zeros(len(df))
    skf = StratifiedKFold(k, shuffle=True, random_state=42)
    for tri, tei in skf.split(X, strat):
        clf = make_clf(model)
        clf.fit(X[tri], y[tri])
        oof[tei] = clf.predict_proba(X[tei])[:, 1]
    return report(y, oof, df["cohort"].values)


def cv_regress(df, X, model, k=5):
    y = df["anemia"].values; hgb = df["hgb"].values; sexM = df["sex_M"].values
    strat = (df["cohort"] + "_" + df["anemia"].astype(str)).values
    oof_h = np.zeros(len(df))
    skf = StratifiedKFold(k, shuffle=True, random_state=42)
    for tri, tei in skf.split(X, strat):
        reg = make_reg(model)
        reg.fit(X[tri], hgb[tri])
        oof_h[tei] = reg.predict(X[tei])
    pred = who_label(oof_h, sexM)  # predicted-anemia from predicted Hgb
    # AUC uses negative predicted Hgb margin as score (lower Hgb -> more anemic)
    cut = np.where(sexM == 1, 13.0, 12.0)
    score = cut - oof_h
    rep = report(y, _minmax(score), df["cohort"].values, thr=0.5)
    rep["overall"].update(_hard(y, pred, df["cohort"].values))
    rep["_hgb_mae"] = float(np.abs(oof_h - hgb).mean())
    return rep


def _minmax(s):
    s = np.asarray(s, float); return (s - s.min()) / (s.max() - s.min() + 1e-9)


def _hard(y, pred, cohort):
    return {"acc_hard": float((pred == y).mean())}


def loco_classify(df, X, model):
    y = df["anemia"].values; coh = df["cohort"].values
    res = {}
    for tr_c, te_c in [("India", "Italy"), ("Italy", "India")]:
        tri = coh == tr_c; tei = coh == te_c
        clf = make_clf(model); clf.fit(X[tri], y[tri])
        proba = clf.predict_proba(X[tei])[:, 1]
        res[f"train_{tr_c}_test_{te_c}"] = metrics(y[tei], proba)
    return res


def loco_regress(df, X, model):
    y = df["anemia"].values; hgb = df["hgb"].values; sexM = df["sex_M"].values; coh = df["cohort"].values
    res = {}
    for tr_c, te_c in [("India", "Italy"), ("Italy", "India")]:
        tri = coh == tr_c; tei = coh == te_c
        reg = make_reg(model); reg.fit(X[tri], hgb[tri])
        ph = reg.predict(X[tei])
        cut = np.where(sexM[tei] == 1, 13.0, 12.0)
        m = metrics(y[tei], _minmax(cut - ph))
        m["acc_hard"] = float((who_label(ph, sexM[tei]) == y[tei]).mean())
        m["hgb_mae"] = float(np.abs(ph - hgb[tei]).mean())
        res[f"train_{tr_c}_test_{te_c}"] = m
    return res


def main():
    df = pd.read_csv(FEAT)
    feat_cols = [c for c in df.columns if c not in DROP]
    X = df[feat_cols].astype(float).values
    print(f"{len(df)} patients, {len(feat_cols)} features\n")

    summary = {"n": len(df), "n_features": len(feat_cols)}
    for model in ("hgb", "rf", "logreg"):
        summary[f"cv_classify_{model}"] = cv_classify(df, X, model)
        summary[f"loco_classify_{model}"] = loco_classify(df, X, model)
    for model in ("hgb", "ridge"):
        summary[f"cv_regress_{model}"] = cv_regress(df, X, model)
        summary[f"loco_regress_{model}"] = loco_regress(df, X, model)

    with open(os.path.join(OUT, "anemia_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def line(tag, r):
        o = r["overall"] if "overall" in r else r
        print(f"{tag:28s} AUC {o['auc']:.3f}  sens {o['sens']:.3f}  spec {o['spec']:.3f}  acc {o['acc']:.3f}")

    print("=== POOLED 5-FOLD CV (overall) ===")
    for m in ("hgb", "rf", "logreg"):
        line(f"classify[{m}]", summary[f"cv_classify_{m}"])
    for m in ("hgb", "ridge"):
        line(f"regress[{m}]->WHO", summary[f"cv_regress_{m}"])
    print("\n=== CV per-cohort (best classifier) ===")
    best = max(("hgb", "rf", "logreg"), key=lambda m: summary[f"cv_classify_{m}"]["overall"]["auc"])
    for c in ("India", "Italy"):
        line(f"classify[{best}] {c}", summary[f"cv_classify_{best}"][c])
    print("\n=== LEAVE-ONE-COHORT-OUT ===")
    for m in ("hgb", "rf", "logreg"):
        r = summary[f"loco_classify_{m}"]
        for k, v in r.items():
            line(f"classify[{m}] {k}", v)
    for m in ("hgb", "ridge"):
        r = summary[f"loco_regress_{m}"]
        for k, v in r.items():
            line(f"regress[{m}] {k}", v)
    print(f"\nwrote {os.path.join(OUT,'anemia_results.json')}")


if __name__ == "__main__":
    main()
