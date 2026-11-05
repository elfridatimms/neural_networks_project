"""
Analiza skupa podataka (FIVES + CHASE_DB1).

1) compute_image_stats()  - za svaku sliku racuna statistike slike i maske -> results/metrics/image_stats.csv
2) plot_*()               - grafovi koji se koriste u izvjestaju -> results/figures/*.png

Pokretanje cijele analize:
    python src/data_analysis.py
(ili korak po korak u notebooks/01_analiza_podataka.ipynb)
"""
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, transform

from config import DATA_DIR, SPLITS_DIR, FIG_DIR, METRICS_DIR, SEED

PATCH = 768          # velicina patcha iz projektnog plana
THIN_REL = 4.0       # "tanka" zila: sirina < 4 promila promjera FOV (~7.5 px na FIVES-u, ~3.5 px na CHASE-u)
sns.set_theme(style="whitegrid", font_scale=0.9)
DISEASE_ORDER = ["normalno", "AMD", "DR", "glaukom"]


# ----------------------------------------------------------------------------- ucitavanje
def load_split():
    return pd.read_csv(SPLITS_DIR / "split.csv")


def load_image(rel_path):
    return np.array(Image.open(DATA_DIR / rel_path).convert("RGB"))


def load_mask(rel_path):
    m = np.array(Image.open(DATA_DIR / rel_path))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def fov_mask(img):
    """
    Vidno polje (krug fotografije): pikseli svjetliji od crne pozadine (pozadina je <= 4).
    Zadrzavamo samo najvecu komponentu, da tekst/vodeni zig u kutu ne ude u FOV.
    """
    fov = img.max(axis=2) > 4
    fov = ndi.binary_opening(fov, structure=morphology.disk(3))
    lab, n = ndi.label(fov)
    if n > 1:
        fov = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndi.binary_fill_holes(fov)


# ----------------------------------------------------------------------------- statistike
def _stats_for_row(row):
    img = load_image(row["image"])
    raw_mask = np.array(Image.open(DATA_DIR / row["mask"]))
    mask = load_mask(row["mask"])
    fov = fov_mask(img)
    h, w = mask.shape
    fov_diam = 2 * np.sqrt(fov.sum() / np.pi)
    green = img[..., 1].astype(np.float32)

    # sirina zila: udaljenost od ruba maske na skeletu (sredini zile) * 2
    skel = morphology.skeletonize(mask)
    dist = ndi.distance_transform_edt(mask)
    widths = 2 * dist[skel]
    rel_widths = widths / fov_diam * 1000  # u promilima promjera vidnog polja

    labels = measure.label(mask, connectivity=2)
    comp_sizes = np.bincount(labels.ravel())[1:]

    # ostrina: varijanca Laplaciana na zelenom kanalu (smanjeno na 1024 radi usporedivosti)
    g_small = transform.resize(green, (1024, 1024), anti_aliasing=True, preserve_range=True)
    sharpness = filters.laplace(g_small)[transform.resize(fov, (1024, 1024)) > 0.5].var()

    # lokalni (Weberov) kontrast zila u zelenom kanalu: zila vs. prsten pozadine oko nje
    # (prsten sirine 1 % promjera FOV-a), > 0 znaci da su zile tamnije od okoline
    ring = (ndi.distance_transform_edt(~mask) <= 0.01 * fov_diam) & ~mask & fov
    vessel_contrast = (green[ring].mean() - green[mask].mean()) / (green[ring].mean() + 1e-6)

    # vodeni zig / tekst u gornjem lijevom kutu (izvan FOV-a)
    corner = img[:80, :80].max(axis=2)
    corner_text = bool((corner > 60).sum() > 20)

    # patchevi 768x768 (bez preklapanja) - koliki udio zila ima svaki
    patch_fracs = []
    for y in range(0, h - PATCH + 1, PATCH):
        for x in range(0, w - PATCH + 1, PATCH):
            patch_fracs.append(mask[y:y + PATCH, x:x + PATCH].mean())

    # mali "otisak" maske za trazenje duplikata (maske su puno specificnije od samih slika)
    thumb = transform.resize(mask.astype(np.float32), (64, 64), anti_aliasing=True) > 0.1

    return {
        "id": row["id"],
        "height": h, "width": w,
        "img_mask_same_size": img.shape[:2] == mask.shape,
        "mask_unique_values": len(np.unique(raw_mask)),
        "fov_frac": fov.mean(),
        "fov_diam_px": fov_diam,
        "mean_R": img[..., 0][fov].mean(), "mean_G": img[..., 1][fov].mean(), "mean_B": img[..., 2][fov].mean(),
        "std_G": green[fov].std(),
        "vessel_frac_img": mask.mean(),
        "vessel_frac_fov": mask[fov].mean(),
        "vessel_outside_fov": (mask & ~fov).sum() / max(mask.sum(), 1),
        "skeleton_len_rel": skel.sum() / fov_diam,
        "width_mean_px": widths.mean(),
        "width_median_px": np.median(widths),
        "width_max_px": widths.max(),
        "thin_frac": (rel_widths < THIN_REL).mean(),
        "n_components": len(comp_sizes),
        "largest_comp_frac": comp_sizes.max() / comp_sizes.sum(),
        "small_comp_frac": comp_sizes[comp_sizes < 50].sum() / comp_sizes.sum(),
        "sharpness": sharpness,
        "vessel_contrast": vessel_contrast,
        "corner_text": corner_text,
        "patch_fracs": patch_fracs,
        "widths_rel_sample": np.random.default_rng(0).choice(rel_widths, size=min(2000, len(rel_widths)), replace=False),
        "thumb": thumb.ravel(),
    }


def compute_image_stats(split, processes=8, use_cache=True):
    """Statistike za sve slike (~4 min na 8 jezgri). Rezultat se sprema i kasnije ucitava iz cachea."""
    cache = METRICS_DIR / "image_stats_cache.pkl"
    if use_cache and cache.exists():
        stats, extra = pd.read_pickle(cache)
        return stats, extra
    rows = [r for _, r in split.iterrows()]
    with Pool(processes) as pool:
        res = pool.map(_stats_for_row, rows, chunksize=4)
    extra = {r["id"]: {k: r.pop(k) for k in ("patch_fracs", "widths_rel_sample", "thumb")} for r in res}
    stats = split.merge(pd.DataFrame(res), on="id")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(METRICS_DIR / "image_stats.csv", index=False)
    pd.to_pickle((stats, extra), cache)
    return stats, extra


def find_near_duplicates(stats, extra, thr=0.5):
    """
    Parovi slika s vrlo slicnim maskama zila: IoU smanjenih maski (64x64) > thr.
    Nesrodne mreznice imaju IoU ~0.1-0.2; isto oko snimljeno vise puta daje puno veci IoU.
    """
    ids = stats["id"].tolist()
    T = np.stack([extra[i]["thumb"].ravel() for i in ids]).astype(np.float32)
    inter = T @ T.T
    area = T.sum(1)
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-6)
    np.fill_diagonal(iou, 0)
    a, b = np.where(np.triu(iou) > thr)
    pairs = pd.DataFrame({"id_a": [ids[k] for k in a], "id_b": [ids[k] for k in b], "iou": iou[a, b]})
    sp = stats.set_index("id")["split"]
    pairs["split_a"] = pairs.id_a.map(sp); pairs["split_b"] = pairs.id_b.map(sp)
    return pairs.sort_values("iou", ascending=False), iou


def summary_table(stats):
    cols = {"vessel_frac_fov": "Udio zila u FOV", "width_mean_px": "Sr. sirina zile [px]",
            "thin_frac": "Udio tankih zila", "vessel_contrast": "Kontrast zila (G)",
            "sharpness": "Ostrina (var. Laplace)", "n_components": "Br. komponenti"}
    out = []
    for name, sub in [("FIVES", stats[stats.dataset == "FIVES"]), ("CHASE_DB1", stats[stats.dataset == "CHASE_DB1"])]:
        for c, label in cols.items():
            v = sub[c]
            out.append({"skup": name, "znacajka": label, "prosjek": v.mean(), "medijan": v.median(),
                        "min": v.min(), "max": v.max(), "std": v.std()})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- grafovi
def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_examples(split):
    """Po jedan primjer za svaku bolest iz FIVES-a i jedan iz CHASE_DB1: slika + maska."""
    rng = np.random.default_rng(SEED)
    picks = []
    for d in DISEASE_ORDER:
        sub = split[(split.dataset == "FIVES") & (split.disease == d) & (split.q_blur == 1)]
        picks.append((f"FIVES - {d}", sub.iloc[rng.integers(len(sub))]))
    picks.append(("CHASE_DB1", split[split.dataset == "CHASE_DB1"].iloc[0]))
    fig, ax = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.2))
    for k, (title, r) in enumerate(picks):
        ax[0, k].imshow(load_image(r["image"]))
        ax[1, k].imshow(load_mask(r["mask"]), cmap="gray")
        ax[0, k].set_title(title)
        for a in ax[:, k]:
            a.axis("off")
    _save(fig, "fig_examples.png")


def plot_distribution(split):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ct = pd.crosstab(split["dataset"] + " / " + split["disease"], split["split"])[["train", "val", "test"]]
    ct.plot.barh(stacked=True, ax=ax[0], color=["#4C72B0", "#DD8452", "#55A868"])
    ax[0].set_xlabel("broj slika"); ax[0].set_ylabel(""); ax[0].set_title("Slike po izvoru, bolesti i podskupu")
    q = split[split.dataset == "FIVES"].groupby("disease")[["q_illumination", "q_blur", "q_contrast"]] \
        .apply(lambda d: (1 - d).mean() * 100).loc[DISEASE_ORDER]
    q.columns = ["osvjetljenje/boja", "zamucenje", "nizak kontrast"]
    q.plot.bar(ax=ax[1], rot=0)
    ax[1].set_ylabel("% slika s problemom"); ax[1].set_xlabel(""); ax[1].set_title("FIVES: problemi kvalitete po bolesti")
    _save(fig, "fig_distribution.png")


def plot_vessel_stats(stats):
    stats = stats.copy()
    stats["grupa"] = np.where(stats.dataset == "CHASE_DB1", "CHASE", stats.disease)
    order = DISEASE_ORDER + ["CHASE"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    sns.boxplot(data=stats, x="grupa", y="vessel_frac_fov", order=order, ax=ax[0])
    ax[0].set_title("Udio piksela zila (unutar FOV)"); ax[0].set_ylabel("udio"); ax[0].set_xlabel("")
    sns.boxplot(data=stats, x="grupa", y="thin_frac", order=order, ax=ax[1])
    ax[1].set_title("Udio tankih zila na skeletu"); ax[1].set_ylabel("udio"); ax[1].set_xlabel("")
    sns.boxplot(data=stats, x="grupa", y="vessel_contrast", order=order, ax=ax[2])
    ax[2].set_title("Lokalni kontrast zila (zeleni kanal)"); ax[2].set_ylabel("(okolina - zila) / okolina"); ax[2].set_xlabel("")
    _save(fig, "fig_vessel_stats.png")


def plot_width_hist(stats, extra):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, sub in stats.groupby("dataset"):
        w = np.concatenate([extra[i]["widths_rel_sample"] for i in sub["id"]])
        ax.hist(w, bins=np.linspace(0, 15, 61), density=True, alpha=0.6, label=name)
    ax.axvline(THIN_REL, color="k", ls="--", lw=1, label="prag 'tanke' zile")
    ax.set_xlabel("sirina zile [promil promjera FOV]"); ax.set_ylabel("gustoca"); ax.legend()
    ax.set_title("Raspodjela sirina zila (tocke skeleta)")
    _save(fig, "fig_width_hist.png")


def local_contrast(channel, mask, fov):
    """Weberov kontrast: (srednja vrijednost okoline - srednja vrijednost zile) / okolina."""
    fov_diam = 2 * np.sqrt(fov.sum() / np.pi)
    ring = (ndi.distance_transform_edt(~mask) <= 0.01 * fov_diam) & ~mask & fov
    bg = channel[ring].mean()
    return (bg - channel[mask].mean()) / (bg + 1e-6)


def _channel_contrast_row(r):
    img, m = load_image(r["image"]), load_mask(r["mask"])
    fov = fov_mask(img)
    return [local_contrast(img[..., k].astype(np.float32), m, fov) for k in range(3)]


def plot_channel_contrast(split, n=80):
    """Lokalni kontrast zila u R, G i B kanalu (uzorak slika iz train skupa FIVES-a)."""
    sub = split[(split.split == "train") & (split.dataset == "FIVES")].sample(n, random_state=SEED)
    with Pool(8) as pool:
        vals = np.array(pool.map(_channel_contrast_row, [r for _, r in sub.iterrows()]))
    df = pd.DataFrame(vals, columns=["R", "G", "B"])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    sns.boxplot(data=df, ax=ax, palette=["#C44E52", "#55A868", "#4C72B0"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("lokalni kontrast zila"); ax.set_title(f"Kontrast zila po kanalu (n={n})")
    _save(fig, "fig_channel_contrast.png")
    return df


def preprocess_variants(img):
    """Varijante predobrade iz eksperimenta 2 (za prikaz)."""
    g = img[..., 1]
    clahe = exposure.equalize_adapthist(g, clip_limit=0.01, kernel_size=64)
    gamma = exposure.adjust_gamma(clahe, 1.2)
    return {"RGB": img, "zeleni kanal": g, "CLAHE (G)": clahe, "CLAHE + gamma": gamma}


def plot_preprocessing(split, image_id):
    r = split.set_index("id").loc[image_id]
    img = load_image(r["image"])
    variants = preprocess_variants(img)
    h = img.shape[0]
    y0, x0, s = int(h * 0.35), int(h * 0.55), int(h * 0.15)  # izrez s tankim zilama
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for k, (name, v) in enumerate(variants.items()):
        cmap = None if v.ndim == 3 else "gray"
        ax[0, k].imshow(v, cmap=cmap); ax[0, k].set_title(name)
        ax[1, k].imshow(v[y0:y0 + s, x0:x0 + s], cmap=cmap)
        for a in ax[:, k]:
            a.axis("off")
    _save(fig, "fig_preprocessing.png")


def plot_patches(stats, extra):
    fr = np.concatenate([extra[i]["patch_fracs"] for i in stats[stats.dataset == "FIVES"]["id"]])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(fr, bins=40, color="#4C72B0")
    ax.set_xlabel("udio piksela zila u patchu 768x768"); ax.set_ylabel("broj patcheva")
    ax.set_title(f"FIVES patchevi (bez preklapanja), n={len(fr)}")
    _save(fig, "fig_patches.png")
    return fr


def plot_correlation(stats):
    cols = {"vessel_frac_fov": "udio zila", "thin_frac": "udio tankih", "width_mean_px": "sr. sirina",
            "n_components": "br. komponenti", "small_comp_frac": "udio sitnih komp.",
            "vessel_contrast": "kontrast zila", "sharpness": "ostrina", "mean_G": "sr. G", "std_G": "std G",
            "q_blur": "Blur ok", "q_illumination": "IC ok", "q_contrast": "LC ok"}
    c = stats[stats.dataset == "FIVES"][list(cols)].rename(columns=cols).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 5.6))
    sns.heatmap(c, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Spearmanova korelacija znacajki (FIVES)")
    _save(fig, "fig_correlation.png")
    return c


def pick_easy_hard(stats):
    """Lako: visok kontrast zila, ostro, bez problema kvalitete. Tesko: nizak kontrast / mutno / puno tankih zila."""
    f = stats[stats.dataset == "FIVES"].copy()
    ok = (f.q_blur == 1) & (f.q_illumination == 1) & (f.q_contrast == 1)
    f["score"] = f["vessel_contrast"].rank(pct=True) + f["sharpness"].rank(pct=True) - f["thin_frac"].rank(pct=True)
    easy = f[ok].sort_values("score", ascending=False).head(2)
    hard = f.sort_values("score").head(2)
    return easy, hard


def plot_easy_hard(easy, hard):
    rows = [("lako", r) for _, r in easy.iterrows()] + [("tesko", r) for _, r in hard.iterrows()]
    fig, ax = plt.subplots(2, len(rows), figsize=(3 * len(rows), 6.2))
    for k, (lab, r) in enumerate(rows):
        ax[0, k].imshow(load_image(r["image"]))
        ax[0, k].set_title(f"{lab}: {r['disease']}\nkontrast={r['vessel_contrast']:.2f}", fontsize=9)
        ax[1, k].imshow(load_mask(r["mask"]), cmap="gray")
        for a in ax[:, k]:
            a.axis("off")
    _save(fig, "fig_easy_hard.png")


def plot_split_check(stats):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.kdeplot(data=stats[stats.dataset == "FIVES"], x="vessel_frac_fov", hue="split",
                common_norm=False, ax=ax, hue_order=["train", "val", "test"])
    ax.set_title("Udio zila po podskupovima (FIVES)"); ax.set_xlabel("udio piksela zila u FOV")
    _save(fig, "fig_split_check.png")


def plot_duplicates(split, pairs):
    """Primjeri parova slika istog oka (gotovo iste maske) - mogući izvor curenja podataka."""
    idx = split.set_index("id")
    fig, ax = plt.subplots(len(pairs), 4, figsize=(9, 2.3 * len(pairs)))
    for k, (a, b, iou) in enumerate(pairs):
        for j, i in enumerate([a, b]):
            r = idx.loc[i]
            ax[k, 2 * j].imshow(load_image(r["image"])[::4, ::4])
            ax[k, 2 * j].set_title(f"{i.replace('FIVES_', '')} ({r['orig_split']})", fontsize=8)
            ax[k, 2 * j + 1].imshow(load_mask(r["mask"])[::4, ::4], cmap="gray")
            ax[k, 2 * j + 1].set_title(f"maska, IoU={iou:.2f}", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    _save(fig, "fig_duplicates.png")


def main():
    split = load_split()
    split = split[split.split != "excluded"].reset_index(drop=True)
    stats, extra = compute_image_stats(split)
    print(summary_table(stats).round(3).to_string())
    summary_table(stats).to_csv(METRICS_DIR / "summary_stats.csv", index=False)
    plot_examples(split); plot_distribution(split); plot_vessel_stats(stats)
    plot_width_hist(stats, extra); plot_channel_contrast(split)
    plot_preprocessing(split, "FIVES_train_455_N"); plot_patches(stats, extra)
    plot_correlation(stats); plot_split_check(stats)
    easy, hard = pick_easy_hard(stats); plot_easy_hard(easy, hard)
    dups, _ = find_near_duplicates(stats, extra)
    dups.to_csv(METRICS_DIR / "near_duplicates.csv", index=False)
    plot_duplicates(split, [tuple(r) for r in dups.head(1)[["id_a", "id_b", "iou"]].to_numpy()]
                    + [("FIVES_train_174_D", "FIVES_test_054_D", 0.998), ("FIVES_train_472_N", "FIVES_test_118_G", 0.685)])
    print("Moguci duplikati:", len(dups))


if __name__ == "__main__":
    main()
