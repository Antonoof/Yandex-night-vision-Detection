"""Turning the per-box contrast rows into numbers and figures to argue with."""

import csv
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis.contrast import CSV_COLUMNS  # noqa: E402

logger = logging.getLogger(__name__)

NIGHT, DAY = "night", "daytime"


def write_csv(rows, path):
    """One row per box, for slicing further in a notebook."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("таблица по боксам: %s (%d строк)", path, len(rows))


def _split(rows, key):
    night = np.array([r[key] for r in rows if r["timeofday"] == NIGHT], dtype=float)
    day = np.array([r[key] for r in rows if r["timeofday"] == DAY], dtype=float)
    return night, day


def summarize(rows):
    """Log the comparison the hypothesis actually rests on."""
    night = [r for r in rows if r["timeofday"] == NIGHT]
    day = [r for r in rows if r["timeofday"] == DAY]
    logger.info("боксов: ночь=%d  день=%d", len(night), len(day))

    logger.info("")
    logger.info("%-16s %10s %10s %10s", "метрика", "ночь", "день", "ночь/день")
    logger.info("-" * 50)
    for key, label in [
        ("l_object", "яркость объекта"),
        ("l_background", "яркость фона"),
        ("weber", "контраст Вебера"),
        ("dyn_range", "дин. диапазон"),
        ("rms_contrast", "RMS-контраст"),
    ]:
        n, d = _split(rows, key)
        if not len(n) or not len(d):
            continue
        mn, md = float(np.median(n)), float(np.median(d))
        ratio = mn / md if md else float("nan")
        logger.info("%-16s %10.2f %10.2f %10.2f", label, mn, md, ratio)

    logger.info("")
    logger.info("дин. диапазон (сколько из 255 уровней занимает объект), медиана:")
    logger.info("%-10s %10s %10s", "размер", "ночь", "день")
    for bucket in ("small", "medium", "large"):
        n = np.array(
            [r["dyn_range"] for r in night if r["size_bucket"] == bucket], dtype=float
        )
        d = np.array(
            [r["dyn_range"] for r in day if r["size_bucket"] == bucket], dtype=float
        )
        if len(n) and len(d):
            logger.info(
                "%-10s %10.1f %10.1f", bucket, float(np.median(n)), float(np.median(d))
            )

    tiny = [r for r in night if r["dyn_range"] < 10]
    if night:
        logger.info("")
        logger.info(
            "ночных объектов в пределах <10 уровней яркости: %d из %d (%.1f%%) "
            "- для них растягивать нечего",
            len(tiny),
            len(night),
            100 * len(tiny) / len(night),
        )


def recall_by_contrast(rows, bins=None):
    """Recall as a function of Weber contrast, night vs day.

    This is the decisive plot: same curve for both means contrast explains the
    night-time deficit; a lower night curve at equal contrast means it does not.
    """
    matched = [r for r in rows if r["matched"] is not None]
    if not matched:
        return None
    bins = bins if bins is not None else np.array([-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0])

    out = {}
    for tod, label in ((NIGHT, "ночь"), (DAY, "день")):
        part = [r for r in matched if r["timeofday"] == tod]
        if not part:
            continue
        weber = np.array([r["weber"] for r in part], dtype=float)
        found = np.array([bool(r["matched"]) for r in part], dtype=bool)
        idx = np.digitize(weber, bins) - 1
        centres, recalls, counts = [], [], []
        for b in range(len(bins) - 1):
            sel = idx == b
            if sel.sum() < 30:  # too few to be worth plotting
                continue
            centres.append((bins[b] + bins[b + 1]) / 2)
            recalls.append(float(found[sel].mean()))
            counts.append(int(sel.sum()))
        out[label] = (np.array(centres), np.array(recalls), np.array(counts))

    logger.info("")
    logger.info("recall по корзинам контраста Вебера:")
    for label, (centres, recalls, counts) in out.items():
        logger.info("  %s:", label)
        for c, r, n in zip(centres, recalls, counts):
            logger.info("    контраст ~%+.2f: recall=%.3f  (n=%d)", c, r, n)
    return out


def plot_overview(rows, out_path, recall_curves=None):
    """Four panels: the distributions, and the decisive recall curve."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Контраст объект/фон: ночь против дня", fontsize=15)

    night_w, day_w = _split(rows, "weber")
    ax = axes[0, 0]
    ax.hist(day_w, bins=80, range=(-1, 2), alpha=0.6, label="день", density=True)
    ax.hist(night_w, bins=80, range=(-1, 2), alpha=0.6, label="ночь", density=True)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Контраст Вебера (0 = объект неотличим от фона)")
    ax.set_xlabel("(L_объект − L_фон) / L_фон")
    ax.legend()

    night_d, day_d = _split(rows, "dyn_range")
    ax = axes[0, 1]
    ax.hist(day_d, bins=60, range=(0, 200), alpha=0.6, label="день", density=True)
    ax.hist(night_d, bins=60, range=(0, 200), alpha=0.6, label="ночь", density=True)
    ax.axvline(10, color="crimson", lw=1.2, ls="--", label="10 уровней")
    ax.set_title("Динамический диапазон объекта (из 255 уровней)")
    ax.set_xlabel("p95 − p5 яркости внутри бокса")
    ax.legend()

    ax = axes[1, 0]
    buckets = ("small", "medium", "large")
    width = 0.38
    for offset, (tod, label) in zip((-width / 2, width / 2), ((NIGHT, "ночь"), (DAY, "день"))):
        medians = []
        for b in buckets:
            vals = [r["dyn_range"] for r in rows if r["timeofday"] == tod and r["size_bucket"] == b]
            medians.append(float(np.median(vals)) if vals else 0.0)
        ax.bar(np.arange(len(buckets)) + offset, medians, width, label=label)
    ax.set_xticks(np.arange(len(buckets)), buckets)
    ax.set_title("Динамический диапазон по размеру объекта (медиана)")
    ax.legend()

    ax = axes[1, 1]
    if recall_curves:
        for label, (centres, recalls, counts) in recall_curves.items():
            ax.plot(centres, recalls, marker="o", label=f"{label} (n={counts.sum()})")
        ax.set_title("Recall как функция контраста\nсовпали кривые → дело в контрасте")
        ax.set_xlabel("контраст Вебера")
        ax.set_ylabel("recall")
        ax.set_ylim(0, 1)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "нет весов модели:\nrecall не посчитан", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("график: %s", out_path)
    return out_path
