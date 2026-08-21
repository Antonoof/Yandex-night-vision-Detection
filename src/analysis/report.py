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

# Fixed across every figure: swapping night/day colours between panels is a
# reliable way to make an audience misread a chart.
NIGHT_COLOR = "#3b7dd8"
DAY_COLOR = "#f0932b"


def read_boxes_csv(path):
    """Load boxes.csv back into the row dicts the plotting functions expect.

    csv gives everything back as strings; without this every notebook would
    re-implement the same casting, and one of them would eventually get
    "False" -> bool("False") == True wrong.
    """
    numeric = {
        "l_object": float,
        "l_background": float,
        "weber": float,
        "dyn_range": float,
        "rms_contrast": float,
        "best_iou": float,
        "area_px": int,
        "class_id": int,
        "gt_x1": int,
        "gt_y1": int,
        "gt_x2": int,
        "gt_y2": int,
        "pred_x1": int,
        "pred_y1": int,
        "pred_x2": int,
        "pred_y2": int,
    }
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, cast in numeric.items():
                value = row.get(key)
                row[key] = cast(value) if value not in (None, "") else None
            row["matched"] = {"True": True, "False": False}.get(row.get("matched"))
            rows.append(row)
    logger.info("прочитано боксов: %d (%s)", len(rows), path)
    return rows


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
    ax.hist(day_w, bins=80, range=(-1, 2), alpha=0.6, label="день", density=True, color=DAY_COLOR)
    ax.hist(night_w, bins=80, range=(-1, 2), alpha=0.6, label="ночь", density=True, color=NIGHT_COLOR)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Контраст Вебера (0 = объект неотличим от фона)")
    ax.set_xlabel("(L_объект − L_фон) / L_фон")
    ax.legend()

    night_d, day_d = _split(rows, "dyn_range")
    ax = axes[0, 1]
    ax.hist(day_d, bins=60, range=(0, 200), alpha=0.6, label="день", density=True, color=DAY_COLOR)
    ax.hist(night_d, bins=60, range=(0, 200), alpha=0.6, label="ночь", density=True, color=NIGHT_COLOR)
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
        ax.bar(
            np.arange(len(buckets)) + offset,
            medians,
            width,
            label=label,
            color=NIGHT_COLOR if label == "ночь" else DAY_COLOR,
        )
    ax.set_xticks(np.arange(len(buckets)), buckets)
    ax.set_title("Динамический диапазон по размеру объекта (медиана)")
    ax.legend()

    ax = axes[1, 1]
    if recall_curves:
        for label, (centres, recalls, counts) in recall_curves.items():
            ax.plot(
                centres,
                recalls,
                marker="o",
                label=f"{label} (n={counts.sum()})",
                color=NIGHT_COLOR if label == "ночь" else DAY_COLOR,
            )
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


def localization_summary(rows):
    """Log the IoU comparison: found objects, but how well are they framed."""
    matched = [r for r in rows if r["matched"]]
    if not matched:
        logger.info("нет сопоставленных боксов — локализацию считать не по чему")
        return

    logger.info("")
    logger.info("КАЧЕСТВО ЛОКАЛИЗАЦИИ (IoU среди найденных объектов)")
    logger.info(
        "%-8s %9s %9s %13s %12s", "", "медиана", "среднее", "доля IoU>0.75", "IoU>0.9"
    )
    for tod, label in ((NIGHT, "ночь"), (DAY, "день")):
        vals = np.array(
            [r["best_iou"] for r in matched if r["timeofday"] == tod], dtype=float
        )
        if not len(vals):
            continue
        logger.info(
            "%-8s %9.3f %9.3f %12.1f%% %11.1f%%",
            label,
            float(np.median(vals)),
            float(vals.mean()),
            100 * float((vals > 0.75).mean()),
            100 * float((vals > 0.9).mean()),
        )

    curves = recall_vs_iou_threshold(rows)
    if "ночь" in curves and "день" in curves:
        thr, rec_night = curves["ночь"]
        _, rec_day = curves["день"]
        logger.info("")
        logger.info("отставание ночи по порогам IoU (относительное):")
        for t_, n_, d_ in zip(thr, rec_night, rec_day):
            if d_ > 0:
                logger.info(
                    "  IoU>=%.2f: ночь=%.3f день=%.3f  отставание=%5.1f%%",
                    t_,
                    n_,
                    d_,
                    100 * (d_ - n_) / d_,
                )

    logger.info("")
    logger.info("медиана IoU по размеру объекта:")
    logger.info("%-8s %9s %9s %9s", "размер", "ночь", "день", "разница")
    for bucket in ("small", "medium", "large"):
        n = np.array(
            [r["best_iou"] for r in matched if r["timeofday"] == NIGHT and r["size_bucket"] == bucket],
            dtype=float,
        )
        d = np.array(
            [r["best_iou"] for r in matched if r["timeofday"] == DAY and r["size_bucket"] == bucket],
            dtype=float,
        )
        if len(n) and len(d):
            mn, md = float(np.median(n)), float(np.median(d))
            logger.info("%-8s %9.3f %9.3f %+9.3f", bucket, mn, md, mn - md)


# Signed edge errors are all oriented the same way on purpose: positive means
# the prediction sticks out past the ground truth. Mixing conventions between
# edges would make "the box is inflated" read as zero on average.
EDGE_LABELS = {"left": "лево", "right": "право", "top": "верх", "bottom": "низ"}


def box_errors(rows):
    """Decompose each matched box into placement error and size error.

    IoU says a box is wrong but not *how*. Everything here is normalised by
    the ground-truth size, so a 10-pixel slip on a bus and on a traffic light
    are the same number - otherwise the statistic just re-measures which
    objects happen to be big.

    Args:
        rows (list[dict]): rows from boxes.csv, matched or not; unmatched
            rows and rows without prediction coordinates are dropped.
    Returns:
        dict[str, np.ndarray]: ``dx``/``dy`` centre offset, ``sw``/``sh``
        size ratio, ``left``/``right``/``top``/``bottom`` edge errors, plus
        the ``rows`` that survived, in the same order.
    """
    kept = [
        r
        for r in rows
        if r.get("matched") and r.get("pred_x1") is not None
    ]
    if not kept:
        return {"rows": []}

    gt = np.array(
        [[r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"]] for r in kept], dtype=float
    )
    pr = np.array(
        [[r["pred_x1"], r["pred_y1"], r["pred_x2"], r["pred_y2"]] for r in kept],
        dtype=float,
    )
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    pw, ph = pr[:, 2] - pr[:, 0], pr[:, 3] - pr[:, 1]
    return {
        "rows": kept,
        "dx": ((pr[:, 0] + pr[:, 2]) - (gt[:, 0] + gt[:, 2])) / (2 * gw),
        "dy": ((pr[:, 1] + pr[:, 3]) - (gt[:, 1] + gt[:, 3])) / (2 * gh),
        "sw": pw / gw,
        "sh": ph / gh,
        "left": (gt[:, 0] - pr[:, 0]) / gw,
        "right": (pr[:, 2] - gt[:, 2]) / gw,
        "top": (gt[:, 1] - pr[:, 1]) / gh,
        "bottom": (pr[:, 3] - gt[:, 3]) / gh,
    }


def _iqr(values):
    """Spread that a handful of wild misses cannot inflate."""
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return float("nan")
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def _split_errors(rows):
    """Per-time-of-day error dicts, night first."""
    return {
        label: box_errors([r for r in rows if r["timeofday"] == tod])
        for tod, label in ((NIGHT, "ночь"), (DAY, "день"))
    }


def _spread_row(errors, key, min_n=150):
    """IQR of one error component, night vs day, or None if too few boxes."""
    night, day = errors["ночь"], errors["день"]
    if len(night["rows"]) < min_n or len(day["rows"]) < min_n:
        return None
    a, b = _iqr(night[key]), _iqr(day[key])
    return a, b, (100 * (a / b - 1) if b else float("nan"))


def localization_decomposition(rows, min_n=150):
    """Is the night box *displaced*, or merely *unsteady*?

    A systematic component would show up in the medians - the model bleeding
    boxes out into headlight glow, say, would inflate them at night. A random
    component shows up in the spread only. The two call for different fixes,
    and the medians vs IQR contrast below is what separates them.
    """
    errors = _split_errors(rows)
    if not errors["ночь"]["rows"] or not errors["день"]["rows"]:
        logger.info("нет сопоставленных боксов — раскладывать нечего")
        return

    logger.info("")
    logger.info("РАЗЛОЖЕНИЕ ОШИБКИ ЛОКАЛИЗАЦИИ (доли размера объекта)")
    logger.info("%-26s %9s %9s %9s", "", "ночь", "день", "ночь/день")
    for key, title in (
        ("dx", "смещение центра, x"),
        ("dy", "смещение центра, y"),
        ("sw", "отношение ширин"),
        ("sh", "отношение высот"),
    ):
        med_n = float(np.median(errors["ночь"][key]))
        med_d = float(np.median(errors["день"][key]))
        logger.info("%-26s %9.4f %9.4f", f"медиана {title}", med_n, med_d)
    logger.info("")
    for key, title in (
        ("dx", "смещение центра, x"),
        ("dy", "смещение центра, y"),
        ("sw", "отношение ширин"),
        ("sh", "отношение высот"),
    ):
        spread = _spread_row(errors, key, min_n)
        if spread:
            logger.info("%-26s %9.4f %9.4f %+8.0f%%", f"разброс {title}", *spread)

    logger.info("")
    logger.info("разброс по рёбрам (IQR, + = предсказание выходит наружу):")
    logger.info("%-26s %9s %9s %9s", "ребро", "ночь", "день", "ночь/день")
    for key, label in EDGE_LABELS.items():
        spread = _spread_row(errors, key, min_n)
        if spread:
            logger.info("%-26s %9.4f %9.4f %+8.0f%%", label, *spread)

    for field, title in (("size_bucket", "размеру"), ("class_name", "классу")):
        logger.info("")
        logger.info("разброс смещения по %s (IQR dx / IQR dy):", title)
        logger.info("%-16s %7s %7s %17s %17s", "срез", "n ноч", "n день", "dx", "dy")
        for value in sorted({r[field] for r in rows if r.get("matched")}):
            subset = [r for r in rows if r[field] == value]
            sub_errors = _split_errors(subset)
            dx = _spread_row(sub_errors, "dx", min_n)
            dy = _spread_row(sub_errors, "dy", min_n)
            if not (dx and dy):
                continue
            logger.info(
                "%-16s %7d %7d %17s %17s",
                value,
                len(sub_errors["ночь"]["rows"]),
                len(sub_errors["день"]["rows"]),
                f"{dx[0]:.4f}/{dx[1]:.4f} {dx[2]:+.0f}%",
                f"{dy[0]:.4f}/{dy[1]:.4f} {dy[2]:+.0f}%",
            )


def largest_common_slice(rows, min_n=150):
    """The (class, size) cell with the most night boxes, present in both.

    Controlling for class and size at once costs sample size, so the control
    is run inside one cell rather than across all of them. Picking the cell by
    night count rather than naming it keeps the code honest if the dataset
    changes under it.
    """
    best, best_n = None, 0
    matched = [r for r in rows if r.get("matched")]
    for key in {(r["class_name"], r["size_bucket"]) for r in matched}:
        cell = [
            r for r in matched
            if (r["class_name"], r["size_bucket"]) == key
        ]
        n_night = sum(r["timeofday"] == NIGHT for r in cell)
        n_day = len(cell) - n_night
        if n_night > best_n and n_night >= min_n and n_day >= min_n:
            best, best_n = key, n_night
    return best


def spread_by_contrast(rows, key="dy", edges=(0.0, 0.1, 0.2, 0.35, 0.6, 1.0), min_n=150):
    """Error spread against |Weber contrast|, inside one class/size cell.

    The last confound standing: night boxes could be shakier simply because
    night objects sit at different contrasts. Holding class, size *and*
    contrast fixed answers whether anything is left over - and something is.

    Returns:
        tuple: ``(slice_key, centres, iqr_night, iqr_day, counts_night)``, or
        ``(None, ...)`` when no cell is populated enough.
    """
    cell_key = largest_common_slice(rows, min_n)
    if cell_key is None:
        return None, [], [], [], []
    cell = [
        r for r in rows
        if r.get("matched") and (r["class_name"], r["size_bucket"]) == cell_key
    ]
    centres, night_iqr, day_iqr, counts = [], [], [], []
    for lo, hi in zip(edges, edges[1:]):
        band = [r for r in cell if lo <= abs(r["weber"]) < hi]
        errors = _split_errors(band)
        spread = _spread_row(errors, key, min_n)
        if not spread:
            continue
        centres.append((lo + hi) / 2)
        night_iqr.append(spread[0])
        day_iqr.append(spread[1])
        counts.append(len(errors["ночь"]["rows"]))
    return cell_key, centres, night_iqr, day_iqr, counts


def contrast_control_summary(rows, key="dy", min_n=150):
    """Log the contrast-controlled comparison produced by spread_by_contrast."""
    cell_key, centres, night_iqr, day_iqr, counts = spread_by_contrast(
        rows, key=key, min_n=min_n
    )
    if cell_key is None:
        logger.info("контроль по контрасту: ни одна клетка класс×размер не набрала боксов")
        return
    logger.info("")
    logger.info(
        "КОНТРОЛЬ ПО КОНТРАСТУ: разброс '%s' внутри среза %s/%s",
        key,
        *cell_key,
    )
    logger.info("%-12s %7s %9s %9s %9s", "|weber|", "n ноч", "ночь", "день", "ночь/день")
    for c, n_, d_, cnt in zip(centres, night_iqr, day_iqr, counts):
        logger.info(
            "%-12s %7d %9.4f %9.4f %+8.0f%%",
            f"~{c:.2f}",
            cnt,
            n_,
            d_,
            100 * (n_ / d_ - 1) if d_ else float("nan"),
        )


def recall_vs_iou_threshold(rows, thresholds=None):
    """Recall as a function of the IoU threshold, night vs day.

    The decisive curve for the localization story. A detection problem would
    shift both curves down by a constant; a *localization* problem makes the
    night curve fall away faster as the threshold tightens - which is the same
    thing the journal shows as the gap growing from mAP@50 to mAP@50-95.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)
    out = {}
    for tod, label in ((NIGHT, "ночь"), (DAY, "день")):
        vals = np.array(
            [r["best_iou"] for r in rows if r["timeofday"] == tod], dtype=float
        )
        if not len(vals):
            continue
        out[label] = (thresholds, np.array([(vals >= t).mean() for t in thresholds]))
    return out


def plot_localization(rows, out_path):
    """Four panels arguing that the night gap is about box precision."""
    matched = [r for r in rows if r["matched"]]
    if not matched:
        logger.info("нечего рисовать: нет сопоставленных боксов")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Точность рамок: ночь против дня", fontsize=15)

    ax = axes[0, 0]
    for tod, label, colour in ((DAY, "день", DAY_COLOR), (NIGHT, "ночь", NIGHT_COLOR)):
        vals = [r["best_iou"] for r in matched if r["timeofday"] == tod]
        ax.hist(
            vals, bins=40, range=(0.5, 1.0), alpha=0.6, density=True, label=label, color=colour
        )
    ax.set_title("Распределение IoU среди найденных объектов")
    ax.set_xlabel("IoU с разметкой")
    ax.legend()

    curves = recall_vs_iou_threshold(rows)
    ax = axes[0, 1]
    for label, (thr, rec) in curves.items():
        ax.plot(
            thr,
            rec,
            marker="o",
            label=label,
            color=NIGHT_COLOR if label == "ночь" else DAY_COLOR,
        )
    ax.set_title("Recall при ужесточении порога IoU\n(ночная кривая падает быстрее)")
    ax.set_xlabel("порог IoU")
    ax.set_ylabel("recall")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    buckets = ("small", "medium", "large")
    width = 0.38
    for offset, (tod, label) in zip(
        (-width / 2, width / 2), ((NIGHT, "ночь"), (DAY, "день"))
    ):
        medians = [
            float(
                np.median(
                    [
                        r["best_iou"]
                        for r in matched
                        if r["timeofday"] == tod and r["size_bucket"] == b
                    ]
                    or [0]
                )
            )
            for b in buckets
        ]
        ax.bar(
            np.arange(len(buckets)) + offset,
            medians,
            width,
            label=label,
            color=NIGHT_COLOR if label == "ночь" else DAY_COLOR,
        )
    ax.set_xticks(np.arange(len(buckets)), buckets)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Медиана IoU по размеру объекта")
    ax.legend()

    ax = axes[1, 1]
    if "ночь" in curves and "день" in curves:
        thr, rec_night = curves["ночь"]
        _, rec_day = curves["день"]
        # Relative, not absolute: in absolute terms the gap peaks and then
        # falls simply because both curves run into the floor near IoU 0.95.
        # Relative lag keeps rising, which is the honest statement.
        lag = 100 * np.where(rec_day > 0, (rec_day - rec_night) / rec_day, np.nan)
        ax.plot(thr, lag, marker="o", color="crimson")
        ax.fill_between(thr, 0, lag, alpha=0.2, color="crimson")
        ax.set_title(
            "Относительное отставание ночи\nчем строже рамка, тем сильнее отстаём"
        )
        ax.set_xlabel("порог IoU")
        ax.set_ylabel("(день − ночь) / день, %")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("график: %s", out_path)
    return out_path


def plot_decomposition(rows, out_path):
    """Four panels separating a displaced box from an unsteady one."""
    errors = _split_errors(rows)
    if not errors["ночь"]["rows"] or not errors["день"]["rows"]:
        logger.info("нечего рисовать: нет сопоставленных боксов")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Ночная рамка не смещена — она дрожит", fontsize=15
    )

    # Panel 1: the distributions themselves. Same centre, wider night curve -
    # everything else on this figure is a way of measuring that picture.
    ax = axes[0, 0]
    shoulders = []
    for label, colour in (("день", DAY_COLOR), ("ночь", NIGHT_COLOR)):
        heights, edges_, _ = ax.hist(
            errors[label]["dy"],
            bins=80,
            range=(-0.3, 0.3),
            density=True,
            histtype="step",
            lw=2,
            label=f"{label} (медиана {np.median(errors[label]['dy']):+.3f})",
            color=colour,
        )
        # Box coordinates are integers, so a large share of predictions land
        # exactly on the ground truth and pile into the central bin. Left in,
        # that single spike is three times the height of everything else and
        # flattens the very comparison this panel exists to show.
        shoulders.append(heights[np.abs(edges_[:-1] + 0.5 * np.diff(edges_)) > 0.01].max())
    ax.set_ylim(0, 1.25 * max(shoulders))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Смещение центра по вертикали\nв долях высоты объекта")
    ax.set_xlabel("(центр предсказания − центр разметки) / высота")
    ax.legend()

    # Panel 2: spread by size. The relative night penalty grows with the
    # object, which is the opposite of where the small-object work went.
    ax = axes[0, 1]
    buckets = [
        b for b in ("small", "medium", "large")
        if _spread_row(_split_errors([r for r in rows if r["size_bucket"] == b]), "dy")
    ]
    width = 0.38
    for offset, label, colour in (
        (-width / 2, "ночь", NIGHT_COLOR),
        (width / 2, "день", DAY_COLOR),
    ):
        vals = [
            _spread_row(
                _split_errors([r for r in rows if r["size_bucket"] == b]), "dy"
            )[0 if label == "ночь" else 1]
            for b in buckets
        ]
        ax.bar(np.arange(len(buckets)) + offset, vals, width, label=label, color=colour)
    for i, b in enumerate(buckets):
        spread = _spread_row(_split_errors([r for r in rows if r["size_bucket"] == b]), "dy")
        ax.text(i, max(spread[:2]) * 1.03, f"{spread[2]:+.0f}%", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(buckets)), buckets)
    ax.set_title("Разброс смещения (IQR) по размеру объекта")
    ax.set_ylabel("IQR смещения по вертикали")
    ax.legend()

    # Panel 3: which edge. Top and bottom degrade harder than left and right -
    # at night an object is outlined by its own lights, which mark its sides.
    ax = axes[1, 0]
    keys = [k for k in EDGE_LABELS if _spread_row(errors, k)]
    for offset, label, colour in (
        (-width / 2, "ночь", NIGHT_COLOR),
        (width / 2, "день", DAY_COLOR),
    ):
        idx = 0 if label == "ночь" else 1
        ax.bar(
            np.arange(len(keys)) + offset,
            [_spread_row(errors, k)[idx] for k in keys],
            width,
            label=label,
            color=colour,
        )
    for i, k in enumerate(keys):
        ax.text(
            i, max(_spread_row(errors, k)[:2]) * 1.03,
            f"{_spread_row(errors, k)[2]:+.0f}%", ha="center", fontsize=9,
        )
    ax.set_xticks(np.arange(len(keys)), [EDGE_LABELS[k] for k in keys])
    ax.set_ylim(0, max(_spread_row(errors, k)[0] for k in keys) * 1.18)
    ax.set_title("Разброс по рёбрам (IQR)")
    # Lower left: the percentage labels sit above the tallest bars, and the
    # default "best" placement puts the legend straight on top of them.
    ax.legend(loc="lower left")

    # Panel 4: the control. Same class, same size, same contrast - and the
    # night spread is still the higher curve.
    ax = axes[1, 1]
    cell_key, centres, night_iqr, day_iqr, _ = spread_by_contrast(rows)
    if cell_key:
        ax.plot(centres, night_iqr, marker="o", color=NIGHT_COLOR, label="ночь")
        ax.plot(centres, day_iqr, marker="o", color=DAY_COLOR, label="день")
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"Контроль: {cell_key[0]}/{cell_key[1]}, при равном контрасте\n"
            "ночь всё равно выше"
        )
        ax.set_xlabel("|контраст Вебера|")
        ax.set_ylabel("IQR смещения по вертикали")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("график: %s", out_path)
    return out_path


def plot_run_comparison(runs, out_path):
    """Night/day mAP across experiments, plus the gap between them.

    Args:
        runs (list[dict]): ``{"label": str, "night": float, "day": float}``,
            in the order they should appear.
        out_path (str | Path): where to write the figure.
    """
    labels = [r["label"] for r in runs]
    night = np.array([r["night"] for r in runs], dtype=float)
    day = np.array([r["day"] for r in runs], dtype=float)
    gap = 100 * (day - night) / day

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )
    x = np.arange(len(runs))
    width = 0.38
    ax1.bar(x - width / 2, night, width, label="ночь", color=NIGHT_COLOR)
    ax1.bar(x + width / 2, day, width, label="день", color=DAY_COLOR)
    for xi, (n, d) in enumerate(zip(night, day)):
        ax1.text(xi - width / 2, n + 0.004, f"{n:.3f}", ha="center", fontsize=9)
        ax1.text(xi + width / 2, d + 0.004, f"{d:.3f}", ha="center", fontsize=9)
    ax1.set_ylabel("mAP@50-95")
    ax1.set_title("Ночной и дневной mAP по экспериментам")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(x, gap, marker="o", color="crimson")
    span = max(gap.max() - gap.min(), 1.0)
    for xi, g in enumerate(gap):
        ax2.text(xi, g + 0.08 * span, f"{g:.1f}%", ha="center", fontsize=9)
    ax2.set_ylim(gap.min() - 0.25 * span, gap.max() + 0.35 * span)
    ax2.set_ylabel("разрыв, %")
    ax2.set_title("Разрыв ночь/день, (день − ночь) / день")
    ax2.set_xticks(x, labels, rotation=20, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("график: %s", out_path)
    return out_path
