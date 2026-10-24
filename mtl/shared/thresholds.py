"""The epochs-to-threshold metric: when did a run first reach a given accuracy.

ONE definition, shared by the transfer sweep and by whatever backfills these fields
into already-trained runs. The numbers only mean anything if a fine-tuned cell and
the from-scratch run it is compared against were computed the same way, down to the
``>=`` and the epoch numbering -- so both sides import this rather than reimplement.

RECOVERED, 2026-10-20. These two functions lived in
``foundation_model/backfill_epochs_to_threshold.py``, which was never committed and
is no longer on disk; only a stale ``__pycache__`` .pyc survived, which is why
``finetune_flat_transfer.py`` had been failing at import. The source below was
reconstructed from that bytecode and verified against it on 3,000 random cases plus
ten thresholds -- exact match, including the ``int()`` cast and the None returns.
"""


def field_names(threshold):
    """0.8 -> ('epochs_to_80', 'reached_80'); 0.95 -> ('epochs_to_95', 'reached_95')."""
    pct = f"{threshold * 100:g}".replace(".", "_")
    return f"epochs_to_{pct}", f"reached_{pct}"


def crossing_epoch(acc, eval_epochs, threshold):
    """First epoch at which acc >= threshold, or None. Mirrors epochs_to_first_perfect.

    `eval_epochs` maps a position in `acc` to a real epoch number, for runs scored on
    an --eval-interval grid; pass None when `acc` is per-epoch and the epoch is just
    the 1-based index.
    """
    for i, v in enumerate(acc):
        if v >= threshold:
            if eval_epochs is not None:
                return int(eval_epochs[i])
            return i + 1
    return None


__all__ = ["crossing_epoch", "field_names"]
