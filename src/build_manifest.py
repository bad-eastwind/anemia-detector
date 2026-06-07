"""Build manifest.csv for the Eyes-defy-anemia dataset.

Per patient: cohort, patient id, image path, the 3 mask paths (blank if missing),
Hgb (g/dL), sex, age, and the WHO anemia label.

WHO Hgb cutoffs (sea level, g/dL), all subjects here are adults (age >= 19):
  - Men  (>=15y): anemic if Hgb < 13.0
  - Women(>=15y, non-pregnant): anemic if Hgb < 12.0
Pregnancy status is unknown for this dataset -> treat all women as non-pregnant
(standard for this benchmark).
"""
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_io import find_patient_files
from paths import DATA_ROOT as ROOT, MANIFEST as OUT


def who_anemia(hgb, sex):
    if pd.isna(hgb):
        return pd.NA
    cutoff = 13.0 if str(sex).strip().upper().startswith("M") else 12.0
    return int(hgb < cutoff)


def parse_hgb(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().replace(",", ".")
    if s in ("", "_", "-", "nan"):
        return pd.NA
    try:
        return float(s)
    except ValueError:
        return pd.NA


def load_meta(cohort):
    f = os.path.join(ROOT, cohort, f"{cohort}.xlsx")
    df = pd.read_excel(f).iloc[:, :4]
    df.columns = ["Number", "Hgb", "Gender", "Age"]
    df["Number"] = df["Number"].astype(int)
    df["Hgb"] = df["Hgb"].map(parse_hgb)
    df["Gender"] = df["Gender"].astype(str).str.strip().str.upper().str[0]
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df.set_index("Number")


def main():
    rows = []
    for cohort in ("India", "Italy"):
        meta = load_meta(cohort)
        cdir = os.path.join(ROOT, cohort)
        for d in sorted(os.listdir(cdir), key=lambda x: (not x.isdigit(), x)):
            p = os.path.join(cdir, d)
            if not (os.path.isdir(p) and d.isdigit()):
                continue
            pid = int(d)
            fl = find_patient_files(p)
            m = meta.loc[pid] if pid in meta.index else None
            hgb = m["Hgb"] if m is not None else pd.NA
            sex = m["Gender"] if m is not None else pd.NA
            age = m["Age"] if m is not None else pd.NA
            rows.append({
                "cohort": cohort,
                "patient": pid,
                "uid": f"{cohort}_{pid}",
                "image": os.path.relpath(fl.get("image", ""), ROOT) if fl.get("image") else "",
                "palpebral": os.path.relpath(fl["palpebral"], ROOT) if "palpebral" in fl else "",
                "forniceal": os.path.relpath(fl["forniceal"], ROOT) if "forniceal" in fl else "",
                "union": os.path.relpath(fl["union"], ROOT) if "union" in fl else "",
                "hgb": hgb,
                "sex": sex,
                "age": age,
                "anemia": who_anemia(hgb, sex),
                "has_palpebral": int("palpebral" in fl),
                "has_forniceal": int("forniceal" in fl),
                "has_union": int("union" in fl),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    # summary
    print(f"manifest -> {OUT}  ({len(df)} patients)")
    print("\nby cohort:")
    print(df.groupby("cohort").agg(n=("uid", "size"),
                                   anemic=("anemia", lambda s: int(s.sum(skipna=True))),
                                   hgb_missing=("hgb", lambda s: int(s.isna().sum()))).to_string())
    print("\nanemia label counts (excl. missing Hgb):")
    print(df.groupby(["cohort", "anemia"]).size().to_string())
    print("\nsex x anemia:")
    print(pd.crosstab([df.cohort, df.sex], df.anemia, dropna=False).to_string())
    print("\npatients missing a mask:")
    print(df[(df.has_palpebral == 0) | (df.has_forniceal == 0) | (df.has_union == 0)]
          [["uid", "has_palpebral", "has_forniceal", "has_union"]].to_string(index=False))


if __name__ == "__main__":
    main()
