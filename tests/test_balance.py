from src.training.balance import build_balance_spec


def _records():
    return [
        {"name": f"day-{i}.jpg", "timeofday": "daytime", "boxes": [(0, 0, 0, 0, 0)]}
        for i in range(8)
    ] + [
        {"name": f"night-{i}.jpg", "timeofday": "night", "boxes": [(1, 0, 0, 0, 0)]}
        for i in range(2)
    ]


def test_manual_night_and_inverse_class_weights():
    config = {
        "adaptive": {
            "enabled": True,
            "normalization": "dataset_mean",
            "timeofday": {
                "enabled": True,
                "mode": "manual",
                "weights": {"daytime": 1.0, "night": 5.0},
            },
            "classes": {
                "enabled": True,
                "mode": "inverse_frequency",
                "power": 1.0,
                "smoothing": 1.0,
                "normalize": True,
                "min_weight": 0.25,
                "max_weight": 5.0,
            },
        }
    }
    spec = build_balance_spec(_records(), ["common", "rare"], config)

    assert spec.timeofday_weights == {"daytime": 1.0, "night": 5.0}
    assert spec.class_weights[1] > spec.class_weights[0]
    assert spec.normalization_factor == 1.8


def test_automatic_timeofday_ratio():
    config = {
        "adaptive": {
            "enabled": True,
            "timeofday": {
                "enabled": True,
                "mode": "inverse_frequency",
                "power": 1.0,
                "smoothing": 0.0,
                "min_weight": 0.1,
                "max_weight": 10.0,
            },
            "classes": {"enabled": False},
        }
    }
    spec = build_balance_spec(_records(), ["common", "rare"], config)

    assert spec.timeofday_weights["daytime"] == 1.0
    assert spec.timeofday_weights["night"] == 4.0
