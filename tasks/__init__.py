"""The 100-task cognitive battery.

Every task is a peer. There is one module per task, all of them in this package,
all conforming to the same interface (input <= BASE_INPUT_DIM=7, output
OUTPUT_DIM=5), and all registered as their own task_id / subject. Mode variants
(pro/anti, match/nonmatch, +delay, ...) are distinct tasks, as intended.

Two kinds of entry reach the battery, and the difference is how they are built,
not what they are:

    TASKS_MAP          tasks written as a class in this package
    +delay/+seq/+int   variants the modifier engine derives from those

Layout:
    tasks/
        __init__.py   <- this registry (TASKS_MAP, FOUNDATION_TASKS_MAP, TAGS, ...)
        dataset.py    <- CognitiveTask base class + datasets
        modifiers.py  <- TaskModifier and the +delay / +seq / +int engines
        <task>.py     <- one module per task

Run `python -m tasks` to print the tagged battery table.

RL / value-based tasks (bandits, reversal, Daw two-step, IGT) and pure ML-RNN
benchmarks are deliberately excluded from the supervised battery.
"""
# Every task lives in this package, one module per task; the registry below is the
# only place their order is decided. `dataset` and `modifiers` are re-exported for
# callers that build their own task lists.
from .anti_reach import AntiReach1D
from .attentional_blink import AttentionalBlink
from .ax_cpt import AXCPT
from .backward_masking import BackwardMasking
from .category_decision import CategoryDecision
from .change_detection import ChangeDetection
from .conditional_reasoning import ConditionalReasoning
from .context_integration import ContextIntegration
from .copy_task import CopyTask
from .dataset import (  # noqa: F401  re-exported for callers
    CognitiveTask,
    HierarchicalTasksDataset,
    MultiTaskDataset,
    collate_fn,
)
from .delay_comparison import DelayComparison
from .delay_match_category import DelayMatchCategory
from .delay_paired_association import DelayPairedAssociation
from .delayed_cue_reversal import DelayedCueReversal
from .delayed_match_to_sample import DelayedMatchToSample
from .delayed_response import DelayedResponse
from .discrimination_opt_out import DiscriminationOptOut
from .dms_distractor import DMSDistractor
from .duration_estimation import DurationEstimation
from .flanker import Flanker
from .flip_flop import FlipFlop
from .gap_detection import GapDetection
from .go_nogo import GoNogo
from .head_direction import HeadDirection
from .hierarchical_reasoning import HierarchicalReasoning
from .interval_discrimination import IntervalDiscrimination
from .interval_reproduction import IntervalReproduction
from .mental_rotation import MentalRotation
from .modifiers import (
    with_delay,
    with_int,
    with_seq,
)
from .multi_sensory_integration import MultiSensoryIntegration
from .noise_cleaner import NoiseCleaner
from .numerosity import Numerosity
from .odd_one_out import OddOneOut
from .oddball_detection import OddballDetection
from .one_two_three_go import OneTwoThreeGo
from .path_integration import PathIntegration1D
from .path_integration_2d import PathIntegration2D
from .perceptual_decision_making import PerceptualDecisionMaking
from .plurality_vote import PluralityVote
from .poisson_clicks import PoissonClicks
from .posner_cueing import PosnerCueing
from .prior_integration import PriorIntegration
from .reaction_time_response import RTResponse
from .relational_match import RelationalMatch
from .serial_order_memory import SerialOrderMemory
from .signal_detection import SignalDetection
from .simon import Simon
from .simultaneity_judgment import SimultaneityJudgment
from .sine_generation import SineGeneration
from .sound_localization import SoundLocalization
from .spatial_wm_retro import SpatialWMRetro
from .sternberg import Sternberg
from .stop_signal import StopSignal
from .stroop import Stroop
from .symbol_category import SymbolCategory
from .task_switch import TaskSwitch
from .temporal_bisection import TemporalBisection
from .temporal_order_judgment import TemporalOrderJudgment
from .visual_search import VisualSearch
from .wm_updating import WMUpdating

# Same phase-duration ranges used by the existing experiments.
duration_params = {
    "context": (5, 10),
    "stimulus": (10, 15),
    "delay": (10, 15),
    "response": (5, 10),
}


def _tag(
    accuracy_type,
    sequence_output=False,
    has_delay=False,
    requires_memory=False,
    requires_rotation=False,
    wm_load=0,
    n_fixed_points=0,
):
    return dict(
        accuracy_type=accuracy_type,
        sequence_output=sequence_output,
        has_delay=has_delay,
        requires_memory=requires_memory,
        requires_rotation=requires_rotation,
        wm_load=wm_load,
        n_fixed_points=n_fixed_points,
    )


# Every task defined by a class in this package, in REGISTRY ORDER.
#
# THE ORDER IS FROZEN. A task's position here is its task_id, and task_ids are
# baked into every checkpoint: the hierarchical p_vector has one ROW per task, the
# flat joint models take a one-hot of this width, and `task_names` in each
# metadata.json is this sequence. Reordering or inserting silently misaligns every
# stored model against the battery. Append only.

TASKS_MAP = {
    "NoiseCleaner": NoiseCleaner(duration_params, noise_level=0.5),
    "DelayPro": DelayedResponse(duration_params, mode="pro"),
    "DelayAnti": DelayedResponse(duration_params, mode="anti"),
    "CatPro": CategoryDecision(duration_params, mode="pro"),
    "CatAnti": CategoryDecision(duration_params, mode="anti"),
    "Match2Sample": DelayedMatchToSample(duration_params, mode="match"),
    "NonMatch2Sample": DelayedMatchToSample(duration_params, mode="nonmatch"),
    "CtxIntMod1": ContextIntegration(duration_params, relevant_modality=1),
    "CtxIntMod2": ContextIntegration(duration_params, relevant_modality=2),
    "CopyTask": CopyTask(duration_params, seq_len=8, num_symbols=4, delay_range=0),
    "GoNogo": GoNogo(duration_params),
    "PerceptualDM": PerceptualDecisionMaking(duration_params),
    "DelayedComparison": DelayComparison(duration_params),
    "DurationPro": DurationEstimation(duration_params, mode="pro"),
    "DurationAnti": DurationEstimation(duration_params, mode="anti"),
    "IntDisc": IntervalDiscrimination(duration_params),
    "MultiSens": MultiSensoryIntegration(duration_params),
    "FlipFlop2": FlipFlop(duration_params, n_bits=2),  # 4 fixed points
    "FlipFlop3": FlipFlop(duration_params, n_bits=3),  # 8 fixed points / bitcode
    "SineGen": SineGeneration(duration_params),  # limit cycle / rotation
    "Sternberg": Sternberg(duration_params, set_size=3),
    "PairedAssoc": DelayPairedAssociation(duration_params),
    "SpatialWMRetro": SpatialWMRetro(duration_params),
    "ChangeDetect": ChangeDetection(duration_params, array_size=3),
    "RTGo": RTResponse(duration_params, mode="pro"),
    "RTAnti": RTResponse(duration_params, mode="anti"),
    "StopSignal": StopSignal(duration_params),
    "AntiReach1D": AntiReach1D(duration_params),
    "Stroop": Stroop(duration_params),
    "Flanker": Flanker(duration_params),
    "Simon": Simon(duration_params),
    "AXCPT": AXCPT(duration_params),
    "TaskSwitch": TaskSwitch(duration_params),
    "IntervalReproduction": IntervalReproduction(duration_params),
    "OneTwoThreeGo": OneTwoThreeGo(duration_params),
    "HeadDirection": HeadDirection(duration_params),
    "PathIntegration1D": PathIntegration1D(duration_params),
    "DMC": DelayMatchCategory(duration_params, mode="match"),
    "DNMC": DelayMatchCategory(duration_params, mode="nonmatch"),
    "DMSDistractor": DMSDistractor(duration_params),
    "HierReason": HierarchicalReasoning(duration_params),
    "Numerosity": Numerosity(duration_params),
    "PathIntegration2D": PathIntegration2D(duration_params),  # 2D continuous attractor
    "TemporalBisection": TemporalBisection(duration_params),  # categorical timing
    "VisualSearch": VisualSearch(duration_params, set_size=4),  # attention / search
    "PosnerCueing": PosnerCueing(duration_params),  # covert spatial attention
    "PoissonClicks": PoissonClicks(duration_params),  # pulse evidence accumulation
    "OddballDetection": OddballDetection(duration_params),  # deviance detection
    "SoundLocalization": SoundLocalization(duration_params),  # auditory azimuth
    "MentalRotation": MentalRotation(duration_params),  # mental imagery
    "BackwardMasking": BackwardMasking(duration_params),  # perception / masking
    "SignalDetection": SignalDetection(duration_params),  # psychophysics / detection
    "SerialOrderMemory": SerialOrderMemory(duration_params),  # memory for order
    "ConditionalReasoning": ConditionalReasoning(duration_params),  # deduction
    "LexicalDecision": SymbolCategory(duration_params),  # symbol-sequence category
    "AttentionalBlink": AttentionalBlink(duration_params),  # temporal attention
    "WMUpdating": WMUpdating(duration_params),  # executive updating
    "RelationalMatch": RelationalMatch(duration_params),  # relations-of-relations
    "ProspectiveMemory": DelayedCueReversal(duration_params),  # cue-gated response reversal
    "TemporalOrderJudgment": TemporalOrderJudgment(duration_params),  # order timing
    "MetacognitiveOptOut": DiscriminationOptOut(duration_params),  # uncertainty opt-out
    "PriorIntegration": PriorIntegration(duration_params),  # prior x likelihood
    "SimultaneityJudgment": SimultaneityJudgment(duration_params),  # coincidence timing
    "GapDetection": GapDetection(duration_params),  # temporal acuity
    "PluralityVote": PluralityVote(duration_params),  # 3-way temporal mode
    "OddOneOut": OddOneOut(duration_params),  # 4-way deviance localization
}

# --- MODIFIER-generated tasks (the scaling engine) ---
# Any task above can be modified; each variant is its own task_id.

# +delay : insert a retention gap before response (reactive -> working memory).
_DELAY_BASES = [
    "GoNogo",
    "PerceptualDM",
    "CatPro",
    "CatAnti",
    "NoiseCleaner",
    "MultiSens",
    "Stroop",
    "Flanker",
    "Simon",
    "TaskSwitch",  # delayed conflict resolution
]
DELAYED_TASKS_MAP = {
    f"{name}+delay": with_delay(TASKS_MAP[name]) for name in _DELAY_BASES
}

# +seq : rotate the angular response over the window (time-varying output).
# +int : rotate the remembered direction across an inserted delay (integration).
# Both apply only to single-target angular bases (loss_channels == [1, 2]).
# +seq/+int append an omega-cue channel, so the base must have a spare channel
# within BASE_INPUT_DIM=7. CtxIntMod1/2 already use 7 channels and are excluded.
_ANGULAR_BASES = [
    "DelayPro",
    "DelayAnti",
    "NoiseCleaner",
    "MultiSens",
    "PerceptualDM",
    "RTGo",
    "RTAnti",
    "SpatialWMRetro",
]
SEQ_TASKS_MAP = {f"{name}+seq": with_seq(TASKS_MAP[name]) for name in _ANGULAR_BASES}
INT_TASKS_MAP = {f"{name}+int": with_int(TASKS_MAP[name]) for name in _ANGULAR_BASES}
# int x seq composition: rotate memory across a delay AND emit a moving output.
# Mod-Cog's headline compositional demand (distinct from int or seq alone).
INTSEQ_TASKS_MAP = {
    f"{name}+int+seq": with_seq(with_int(TASKS_MAP[name])) for name in _ANGULAR_BASES
}

# Task ID -> display name for figures, tables and the paper. The IDs (registry
# keys) stay as they were because saved runs store them in metadata, result
# files and per-task folder names; the display names are also accepted as lookups.
DISPLAY_NAMES = {
    "LexicalDecision": "SymbolCategory",
    "ProspectiveMemory": "DelayedCueReversal",
    "MetacognitiveOptOut": "DiscriminationOptOut",
}
_ID_FROM_DISPLAY = {v: k for k, v in DISPLAY_NAMES.items()}


def _map_base(task, table):
    base, *mods = task.split("+")
    return "+".join([table.get(base, base), *mods])


def display_name(task):
    """Task ID (modifiers kept, e.g. 'ProspectiveMemory+delay') -> display name."""
    return _map_base(task, DISPLAY_NAMES)


def task_id(task):
    """Display name or task ID -> task ID (the registry key)."""
    return _map_base(task, _ID_FROM_DISPLAY)


class _AliasMap(dict):
    """dict that also accepts display names on lookup / membership."""

    def __missing__(self, key):
        tid = task_id(key) if isinstance(key, str) else key
        if tid == key:
            raise KeyError(key)
        return self[tid]

    def __contains__(self, key):
        return dict.__contains__(self, key) or (
            isinstance(key, str) and dict.__contains__(self, task_id(key))
        )

    def get(self, key, default=None):
        return self[key] if key in self else default


FOUNDATION_TASKS_MAP = _AliasMap({
    **TASKS_MAP,
    **DELAYED_TASKS_MAP,
    **SEQ_TASKS_MAP,
    **INT_TASKS_MAP,
    **INTSEQ_TASKS_MAP,
})

# --- analysis covariates per task (editable; modifiers extend these) ---
TAGS = _AliasMap({
    "NoiseCleaner": _tag("angular", requires_memory=True),
    "DelayPro": _tag("angular", has_delay=True, requires_memory=True),
    "DelayAnti": _tag("angular", has_delay=True, requires_memory=True),
    "CatPro": _tag("binary"),
    "CatAnti": _tag("binary"),
    "Match2Sample": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "NonMatch2Sample": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "CtxIntMod1": _tag("angular", requires_memory=True),
    "CtxIntMod2": _tag("angular", requires_memory=True),
    "CopyTask": _tag(
        "copytask",
        sequence_output=True,
        has_delay=True,
        requires_memory=True,
        wm_load=8,
    ),
    "GoNogo": _tag("binary"),
    "PerceptualDM": _tag("angular"),
    "DelayedComparison": _tag(
        "binary", has_delay=True, requires_memory=True, wm_load=1
    ),
    "DurationPro": _tag("binary", requires_memory=True),
    "DurationAnti": _tag("binary", requires_memory=True),
    "IntDisc": _tag("binary", requires_memory=True),
    "MultiSens": _tag("angular"),
    "FlipFlop2": _tag(
        "flipflop",
        sequence_output=True,
        requires_memory=True,
        wm_load=2,
        n_fixed_points=4,
    ),
    "FlipFlop3": _tag(
        "flipflop",
        sequence_output=True,
        requires_memory=True,
        wm_load=3,
        n_fixed_points=8,
    ),
    "SineGen": _tag("copytask", sequence_output=True, requires_rotation=True),
    "Sternberg": _tag("binary", has_delay=True, requires_memory=True, wm_load=3),
    "PairedAssoc": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "SpatialWMRetro": _tag("angular", has_delay=True, requires_memory=True, wm_load=2),
    "ChangeDetect": _tag("binary", has_delay=True, requires_memory=True, wm_load=3),
    "RTGo": _tag("angular"),
    "RTAnti": _tag("angular"),
    "StopSignal": _tag("binary", requires_memory=True),
    "AntiReach1D": _tag("scalar"),
    "Stroop": _tag("binary"),
    "Flanker": _tag("binary"),
    "Simon": _tag("binary"),
    "AXCPT": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "TaskSwitch": _tag("binary"),
    "IntervalReproduction": _tag("scalar", requires_memory=True),
    "OneTwoThreeGo": _tag("scalar", requires_memory=True),
    "HeadDirection": _tag(
        "copytask", sequence_output=True, requires_memory=True, requires_rotation=True
    ),
    "PathIntegration1D": _tag("scalar", requires_memory=True),
    "DMC": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "DNMC": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "DMSDistractor": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "HierReason": _tag("binary"),
    "Numerosity": _tag("scalar", requires_memory=True),
    "PathIntegration2D": _tag(
        "vector", requires_memory=True, requires_rotation=False
    ),
    "TemporalBisection": _tag("binary", requires_memory=True),
    "VisualSearch": _tag("binary", requires_memory=True, wm_load=1),
    "PosnerCueing": _tag("angular", has_delay=True, requires_memory=True),
    "PoissonClicks": _tag("binary", requires_memory=True),
    "OddballDetection": _tag("binary", requires_memory=True),
    "SoundLocalization": _tag("scalar", requires_memory=True),
    "MentalRotation": _tag(
        "binary", has_delay=True, requires_memory=True, requires_rotation=True
    ),
    "BackwardMasking": _tag("binary"),
    "SignalDetection": _tag("binary"),
    "SerialOrderMemory": _tag("binary", requires_memory=True, wm_load=4),
    "ConditionalReasoning": _tag("binary"),
    "LexicalDecision": _tag("binary", requires_memory=True),
    "AttentionalBlink": _tag("binary", requires_memory=True),
    "WMUpdating": _tag("angular", requires_memory=True, wm_load=2),
    "RelationalMatch": _tag("binary", requires_memory=True),
    "ProspectiveMemory": _tag("binary", requires_memory=True),
    "TemporalOrderJudgment": _tag("binary", requires_memory=True),
    "MetacognitiveOptOut": _tag("binary"),
    "PriorIntegration": _tag("angular", requires_memory=True),
    "SimultaneityJudgment": _tag("binary", requires_memory=True),
    "GapDetection": _tag("binary", requires_memory=True),
    "PluralityVote": _tag("argmax", has_delay=True, requires_memory=True, wm_load=3),
    "OddOneOut": _tag("argmax"),  # simultaneous perceptual outlier judgment (no delay)
})

# Auto-derive tags for modifier variants from the base tags (no hand-tuning).
for _name in _DELAY_BASES:
    _t = dict(TAGS[_name])
    _t.update(has_delay=True, requires_memory=True)
    TAGS[f"{_name}+delay"] = _t
for _name in _ANGULAR_BASES:
    # +seq: time-varying rotating output.
    _t = dict(TAGS[_name])
    _t.update(accuracy_type="copytask", sequence_output=True, requires_rotation=True)
    TAGS[f"{_name}+seq"] = _t
    # +int: rotate the memory across an inserted delay (integration).
    _t2 = dict(TAGS[_name])
    _t2.update(has_delay=True, requires_memory=True, requires_rotation=True)
    TAGS[f"{_name}+int"] = _t2
    # +int+seq: composition (rotate memory across delay + moving output).
    _t3 = dict(TAGS[_name])
    _t3.update(
        accuracy_type="copytask",
        sequence_output=True,
        has_delay=True,
        requires_memory=True,
        requires_rotation=True,
    )
    TAGS[f"{_name}+int+seq"] = _t3

assert set(TAGS) == set(FOUNDATION_TASKS_MAP), "TAGS and tasks out of sync"
