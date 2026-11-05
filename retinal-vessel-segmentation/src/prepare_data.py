"""
Priprema podataka: FIVES + CHASE_DB1 -> jedinstveni popis slika i podjela train/val/test.

Pokretanje (iz mape retinal-vessel-segmentation):
    python src/prepare_data.py

Ocekivana struktura podataka (relativno na korijen projekta, vidi config.py):
    data/FIVES/{train,test}/{Original,Ground truth}/*.png
    data/FIVES/Quality Assessment.xlsx
    data/CHASE_DB1_raw/data/{train,test}-00000-of-00001.parquet   (HuggingFace MedOtter/CHASE_DB1)

Izlaz:
    data/CHASE_DB1/{images,masks}/*.png     (CHASE raspakiran iz parquet datoteka)
    splits/index.csv                         (sve slike + metapodaci)
    splits/split.csv                         (index + stupac 'split' = train/val/test)
"""
import io
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import DATA_DIR, SPLITS_DIR, SEED, VAL_FRAC, TEST_FRAC

DISEASE_NAMES = {"A": "AMD", "D": "DR", "G": "glaukom", "N": "normalno"}


def build_fives_index():
    """Popis svih FIVES slika s bolescu i ocjenama kvalitete iz Excel datoteke."""
    root = DATA_DIR / "FIVES"
    quality = pd.read_excel(root / "Quality Assessment.xlsx", sheet_name=None)
    rows = []
    for orig_split, sheet in [("train", "Train"), ("test", "Test")]:
        q = quality[sheet].set_index("Number")
        for img_path in sorted((root / orig_split / "Original").glob("*.png")):
            number, disease = img_path.stem.split("_")
            number = int(number)
            mask_path = root / orig_split / "Ground truth" / img_path.name
            assert mask_path.exists(), f"Nedostaje maska za {img_path}"
            qrow = q.loc[number]
            assert qrow["Disease"] == disease, f"Neslaganje bolesti za {img_path}"
            rows.append({
                "id": f"FIVES_{orig_split}_{number:03d}_{disease}",
                "dataset": "FIVES",
                "group": f"FIVES_{orig_split}_{number:03d}",  # FIVES nema ID pacijenta
                "disease": DISEASE_NAMES[disease],
                "orig_split": orig_split,
                # 1 = slika zadovoljava kriterij kvalitete, 0 = ne zadovoljava
                "q_illumination": int(qrow["IC"]),
                "q_blur": int(qrow["Blur"]),
                "q_contrast": int(qrow["LC"]),
                "image": str(img_path.relative_to(DATA_DIR)),
                "mask": str(mask_path.relative_to(DATA_DIR)),
            })
    return pd.DataFrame(rows)


def extract_chase():
    """CHASE_DB1 dolazi kao parquet; spremamo slike i maske (1st observer) kao PNG."""
    raw = DATA_DIR / "CHASE_DB1_raw" / "data"
    out_img = DATA_DIR / "CHASE_DB1" / "images"
    out_msk = DATA_DIR / "CHASE_DB1" / "masks"
    out_msk2 = DATA_DIR / "CHASE_DB1" / "masks_2ndHO"
    for d in (out_img, out_msk, out_msk2):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for part in ["train", "test"]:
        df = pd.read_parquet(raw / f"{part}-00000-of-00001.parquet")
        for _, r in df.iterrows():
            name = r["image_id"] + ".png"
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(out_img / name)
            for col, out in [("mask", out_msk), ("mask_2ndHO", out_msk2)]:
                m = np.array(Image.open(io.BytesIO(r[col]["bytes"])))
                m = (m > 0).astype(np.uint8) * 255  # binarno 0/255
                if m.ndim == 3:
                    m = m[..., 0]
                Image.fromarray(m).save(out / name)
            rows.append({
                "id": f"CHASE_{r['image_id']}",
                "dataset": "CHASE_DB1",
                "group": f"CHASE_subj{r['subject_id']}",  # oba oka istog djeteta zajedno
                "disease": "normalno",  # zdrava djeca (studija CHASE)
                "orig_split": part,
                "q_illumination": np.nan, "q_blur": np.nan, "q_contrast": np.nan,
                "image": str((out_img / name).relative_to(DATA_DIR)),
                "mask": str((out_msk / name).relative_to(DATA_DIR)),
            })
    return pd.DataFrame(rows)


def find_empty_masks(index):
    """Maske bez ijednog piksela zile (u FIVES-u postoje dvije, na gotovo crnim slikama)."""
    empty = []
    for _, r in index.iterrows():
        m = np.array(Image.open(DATA_DIR / r["mask"]))
        if m.ndim == 3:
            m = m[..., 0]  # neke FIVES maske su RGBA; R kanal nosi masku
        if (m > 127).sum() == 0:
            empty.append(r["id"])
    return empty


def _mask_thumb(rel_path):
    m = Image.open(DATA_DIR / rel_path).convert("L").resize((64, 64), Image.BILINEAR)
    return (np.array(m) > 25).ravel()


def merge_same_eye_groups(index, thr=0.5):
    """
    FIVES nema ID pacijenta, a analiza je pokazala da je isto oko ponekad snimljeno vise puta
    (gotovo identicne maske, ponekad i s razlicitom dijagnozom). Slike cije smanjene maske (64x64)
    imaju IoU > thr spajamo u istu grupu kako ne bi zavrsile u razlicitim podskupovima.
    Za nesrodne slike IoU je tipicno 0.1-0.35.
    """
    with Pool(8) as pool:
        T = np.stack(pool.map(_mask_thumb, index["mask"])).astype(np.float32)
    inter = T @ T.T
    area = T.sum(1)
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-6)
    np.fill_diagonal(iou, 0)

    parent = list(range(len(index)))  # union-find

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    a, b = np.where(np.triu(iou) > thr)
    for i, j in zip(a, b):
        parent[find(i)] = find(j)
    index = index.copy()
    roots = [find(i) for i in range(len(index))]
    groups = index["group"].to_numpy()
    index["group"] = [groups[r] for r in roots]
    print(f"Parova slicnih maski (IoU > {thr}): {len(a)}; spojenih grupa: {index['group'].nunique()}")
    return index


def make_split(index, excluded=()):
    """
    Podjela ~70/15/15 po GRUPAMA (CHASE: dijete, FIVES: isto oko), stratificirano po
    (skup podataka, bolest), tako da su svi izvori i bolesti zastupljeni u svim skupovima.
    Slike iz `excluded` (prazne maske) dobivaju split = 'excluded'.
    """
    rng = np.random.default_rng(SEED)
    index = index.copy()
    index["split"] = ""
    index.loc[index["id"].isin(excluded), "split"] = "excluded"
    keep = index[index["split"] != "excluded"]
    # stratum grupe = (skup, bolest) njenog prvog clana
    gtab = keep.groupby("group")[["dataset", "disease"]].first().reset_index()
    for _, sub in gtab.groupby(["dataset", "disease"]):
        groups = np.array(sorted(sub["group"]))
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * TEST_FRAC))
        n_val = int(round(n * VAL_FRAC))
        test_g = set(groups[:n_test])
        val_g = set(groups[n_test:n_test + n_val])
        for g in groups:
            s = "test" if g in test_g else "val" if g in val_g else "train"
            index.loc[(index["group"] == g) & (index["split"] != "excluded"), "split"] = s
    return index


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    fives = build_fives_index()
    chase = extract_chase()
    index = pd.concat([fives, chase], ignore_index=True)
    index.to_csv(SPLITS_DIR / "index.csv", index=False)

    empty = find_empty_masks(index)
    print("Izbacene slike (prazna maska):", empty)
    index = merge_same_eye_groups(index)
    split = make_split(index, excluded=empty)
    split.to_csv(SPLITS_DIR / "split.csv", index=False)

    print(f"Ukupno slika: {len(index)}")
    print(pd.crosstab([split["dataset"], split["disease"]], split["split"], margins=True))
    # provjera: nijedna grupa (dijete u CHASE-u / isto oko u FIVES-u) nije u vise skupova
    assert split[split.split != "excluded"].groupby("group")["split"].nunique().max() == 1


if __name__ == "__main__":
    main()
