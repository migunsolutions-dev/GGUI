"""Central generated-case/input consistency contract (R-013/R-078/R-099).

Pure Python. Session inputs are copied at Initialize, never inferred from widgets
when Run is pressed. Runtime patch permission must be explicit at the caller;
native case edits conservatively require Initialize until a patch is verified.
"""
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import IntEnum
import copy
import json
import os


class EditImpact(IntEnum):
    DISPLAY_ONLY = 0
    SAFE_RUNTIME_PATCH = 1
    REINITIALIZE_REQUIRED = 2
    NEW_CASE_REQUIRED = 3


DISPLAY_FIELDS = frozenset({'mirrored_view','show_mesh','show_probes','log_scale'})
RUNTIME_PATCH_FIELDS = frozenset({'end_time_s'})


def input_snapshot(inputs):
    values=asdict(inputs) if is_dataclass(inputs) else copy.deepcopy(dict(inputs))
    # Normalize tuples/lists and detach nested mutable dictionaries.
    return json.loads(json.dumps(values,sort_keys=True,allow_nan=False))


def classify_edit(field_name, *, verified_runtime_patch=False):
    if field_name in DISPLAY_FIELDS:
        return EditImpact.DISPLAY_ONLY
    if verified_runtime_patch and field_name in RUNTIME_PATCH_FIELDS:
        return EditImpact.SAFE_RUNTIME_PATCH
    return EditImpact.REINITIALIZE_REQUIRED


@dataclass
class CaseSession:
    case_dir: str = ''
    initialized_inputs: dict = field(default_factory=dict)
    stale: bool = False
    changed_fields: tuple = ()

    def initialized(self,case_dir,inputs):
        self.case_dir=os.path.normcase(os.path.abspath(case_dir))
        self.initialized_inputs=input_snapshot(inputs)
        self.stale=False
        self.changed_fields=()

    def observe(self,inputs,*,verified_runtime_patch=False):
        if not self.case_dir:
            return
        try:
            current=input_snapshot(inputs)
            changed=tuple(sorted(k for k in set(current)|set(self.initialized_inputs)
                         if current.get(k)!=self.initialized_inputs.get(k)
                         and classify_edit(k,verified_runtime_patch=verified_runtime_patch)>=EditImpact.REINITIALIZE_REQUIRED))
        except (TypeError,ValueError):
            changed=('invalid_input',)
        if changed:
            self.stale=True
            self.changed_fields=tuple(sorted(set(self.changed_fields)|set(changed)))

    def runnable(self,case_dir,inputs,*,verified_runtime_patch=False):
        self.observe(inputs,verified_runtime_patch=verified_runtime_patch)
        return bool(self.case_dir and not self.stale and case_dir
                    and self.case_dir==os.path.normcase(os.path.abspath(case_dir)))

    def requirement(self):
        details=', '.join(self.changed_fields)
        return 'Initialize is required before Run.'+(f' Changed inputs: {details}.' if details else '')
