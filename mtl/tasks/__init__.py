"""The 100-task battery. Re-exported from the top-level `tasks` package.

No `__all__` here: the battery's exports are defined by `tasks` itself and change
as tasks are added, so re-declaring them would mean two lists to keep in step.
The redundant `as` aliases are the conventional way to mark a re-export, so the
linter does not read them as unused imports.
"""

from tasks import *
from tasks import FOUNDATION_TASKS_MAP as FOUNDATION_TASKS_MAP
from tasks import TAGS as TAGS
