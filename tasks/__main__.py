"""Print the tagged foundation battery: `python -m tasks`."""

from tasks import (
    DELAYED_TASKS_MAP,
    FOUNDATION_TASKS_MAP,
    INT_TASKS_MAP,
    INTSEQ_TASKS_MAP,
    SEQ_TASKS_MAP,
    TAGS,
    TASKS_MAP,
)

print(f"Tasks (written as a class): {len(TASKS_MAP)}")
print(f"  +delay variants:          {len(DELAYED_TASKS_MAP)}")
print(f"  +seq variants:            {len(SEQ_TASKS_MAP)}")
print(f"  +int variants:            {len(INT_TASKS_MAP)}")
print(f"  +int+seq variants:        {len(INTSEQ_TASKS_MAP)}")
print(f"Total battery:              {len(FOUNDATION_TASKS_MAP)}\n")

hdr = (f"{'task':20s} {'accuracy':9s} {'seq':4s} {'delay':6s} "
       f"{'mem':4s} {'rot':4s} {'wm':3s} {'fp':3s}")
print(hdr)
print("-" * len(hdr))
for name in FOUNDATION_TASKS_MAP:
    t = TAGS[name]
    print(f"{name:20s} {t['accuracy_type']:9s} "
          f"{t['sequence_output']!s:4s} {t['has_delay']!s:6s} "
          f"{t['requires_memory']!s:4s} {t['requires_rotation']!s:4s} "
          f"{t['wm_load']:>3d} {t['n_fixed_points']:>3d}")
