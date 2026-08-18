"""Training-time extensions for the Ultralytics detector."""

from .balance import BalanceSpec, build_balance_spec

__all__ = [
    "AdaptiveDetectionLoss",
    "AdaptiveLossController",
    "BalanceSpec",
    "build_balance_spec",
]


def __getattr__(name):
    """Keep pure balance utilities importable before torch is installed."""
    if name in {"AdaptiveDetectionLoss", "AdaptiveLossController"}:
        from .adaptive_loss import AdaptiveDetectionLoss, AdaptiveLossController

        return {
            "AdaptiveDetectionLoss": AdaptiveDetectionLoss,
            "AdaptiveLossController": AdaptiveLossController,
        }[name]
    raise AttributeError(name)
