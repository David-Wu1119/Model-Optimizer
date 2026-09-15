# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile a quantize config into an ordered list of scoped calibration stages."""

import fnmatch
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch.nn as nn

from .config import AlgoCfgEntry, QuantizeAlgorithmConfig, QuantizeConfig

__all__ = [
    "ALL_WRITABLE_TOKENS",
    "AlgoCapabilities",
    "AlgoCfgValidationError",
    "AlgoStage",
    "CalibrationPlan",
    "capabilities_for",
    "compile_algo_cfg",
    "describe_plan",
    "plan_hash",
    "stage_predicate",
]


class AlgoCfgValidationError(ValueError):
    """Raised when an ``algo_cfg`` cannot be lowered into a valid plan."""


# --- Capabilities ---


@dataclass(frozen=True)
class AlgoCapabilities:
    """What one calibration algorithm reads, writes and assumes."""

    #: Writes *every* quantizer of each linear it touches, not one quantizer at a time.
    writes_whole_module: bool
    #: Role this algorithm *improves*. Narrower than what it writes: weight-side algorithms
    #: also seed input amax via an internal `max_calibrate`, which `produces` records.
    optimizes: str
    #: Tokens read. ``weight`` and ``acts`` are always available.
    consumes: frozenset[str] = frozenset()
    #: Tokens written. An upper bound: may write less on a given model, never more.
    produces: frozenset[str] = frozenset()
    #: Tokens whose presence makes this algorithm incorrect: ``awq_lite`` folds a scale into
    #: the weight assuming an unsmoothed start, so it conflicts with ``pre_quant_scale``.
    conflicts_with: frozenset[str] = frozenset()
    #: Honours the ``should_process`` write-mask. One that does not must never be scoped.
    honors_write_mask: bool = True


WEIGHT_AMAX = "weight_amax"
INPUT_AMAX = "input_amax"
PRE_QUANT_SCALE = "pre_quant_scale"
WEIGHT = "weight"

#: Conservative default for an algorithm that declares nothing: over-reporting conflicts is
#: the safe direction.
ALL_WRITABLE_TOKENS = frozenset({WEIGHT, WEIGHT_AMAX, INPUT_AMAX, PRE_QUANT_SCALE})


def capabilities_for(algo: str | None, cfg: dict | None = None) -> AlgoCapabilities | None:
    """Capabilities of ``algo``, read off its calibrate-mode descriptor."""
    if algo is None:
        return None
    # Imported lazily: `mode` imports this module while the package is still initializing.
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    descriptor = CalibrateModeRegistry.get(BaseCalibrateModeDescriptor._get_mode_name(algo))
    if descriptor is None:
        return None
    return type(descriptor).capabilities_for_cfg(cfg or {})


# --- Stages ---


@dataclass(frozen=True)
class AlgoStage:
    """One algorithm applied to one scope — the unit of work in a calibration plan."""

    algo: str | None
    cfg: dict  # kwargs for the algorithm, including its "method" key
    scope: str  # the glob
    selector: str  # "module_name" | "quantizer_name"
    order: int  # position within its entry's pipeline
    entry: int  # which algo_cfg entry it came from (-1 = the `algorithm` fallback)
    # Scopes this stage must NOT touch. The fallback's coverage is a complement, so it cannot
    # be a glob; globs-minus-globs keeps the plan a pure function of the config (rank-identical).
    exclude: tuple[tuple[str, str], ...] = ()

    @property
    def capabilities(self) -> AlgoCapabilities | None:
        """Declared capabilities of this stage's algorithm, or ``None`` if undeclared."""
        return capabilities_for(self.algo, self.cfg)

    def key(self) -> tuple:
        """Execution-relevant identity, used for the plan hash."""
        return (
            self.algo,
            tuple(sorted(self.cfg.items(), key=str)),
            self.scope,
            self.selector,
            self.order,
            tuple(sorted(self.exclude)),
        )

    def __str__(self) -> str:
        extra = {k: v for k, v in self.cfg.items() if k != "method"}
        extra_s = f" {extra}" if extra else ""
        excl = f" minus {[g for _, g in self.exclude]}" if self.exclude else ""
        return (
            f"{self.algo or 'none'} @ {self.selector}={self.scope!r}{excl} (#{self.order}){extra_s}"
        )


CalibrationPlan = list[AlgoStage]


# --- Target resolution ---


@dataclass
class _ModelIndex:
    """Names of the things a scope can select, read once from the model structure."""

    linears: list[str] = field(default_factory=list)
    quantizers: list[str] = field(default_factory=list)
    quantizers_of: dict[str, list[str]] = field(default_factory=dict)  # linear -> quantizers
    parent_of: dict[str, str] = field(default_factory=dict)  # quantizer -> linear


#: Scoped to one pure computation rather than global: the index is only valid while nothing
#: mutates the module tree.
_CACHED_INDEX: tuple[nn.Module, "_ModelIndex"] | None = None


@contextmanager
def _reusing_model_index(model: nn.Module) -> Iterator[None]:
    global _CACHED_INDEX
    previous = _CACHED_INDEX
    _CACHED_INDEX = (model, _build_model_index(model))
    try:
        yield
    finally:
        _CACHED_INDEX = previous


def _index_model(model: nn.Module) -> _ModelIndex:
    """Cached structural index; see :func:`_reusing_model_index`."""
    if _CACHED_INDEX is not None and _CACHED_INDEX[0] is model:
        return _CACHED_INDEX[1]
    return _build_model_index(model)


def _build_model_index(model: nn.Module) -> _ModelIndex:
    # Imported lazily: `mode` imports this module while `modelopt.torch.quantization` is
    # still initializing, and `.nn` pulls in the quantized-tensor backends.
    from .nn import SequentialQuantizer, TensorQuantizer
    from .utils import is_quantized_linear

    index = _ModelIndex()
    for name, module in model.named_modules():
        if is_quantized_linear(module):
            index.linears.append(name)
            index.quantizers_of[name] = []
        elif isinstance(module, TensorQuantizer | SequentialQuantizer):
            index.quantizers.append(name)
    for q in index.quantizers:
        # Nearest enclosing linear, not the direct parent: a SequentialQuantizer (W4A8,
        # INT4-AWQ) nests levels as `<linear>.weight_quantizer.0`, a grandchild.
        parts = q.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            ancestor = ".".join(parts[:depth])
            if ancestor in index.quantizers_of:
                index.quantizers_of[ancestor].append(q)
                index.parent_of[q] = ancestor
                break
    return index


def resolve_targets(model: nn.Module, scope: str, selector: str) -> tuple[set[str], set[str]]:
    """Resolve a scope into ``(module names, quantizer names)``."""
    index = _index_model(model)
    if selector == "module_name":
        modules = {n for n in index.linears if fnmatch.fnmatch(n, scope)}
        quantizers = {q for m in modules for q in index.quantizers_of[m]}
    else:
        quantizers = {n for n in index.quantizers if fnmatch.fnmatch(n, scope)}
        modules = {index.parent_of[q] for q in quantizers if q in index.parent_of}
    return modules, quantizers


#: Which quantizer a state token lives on.
TOKEN_ROLE: dict[str, str] = {
    "weight": "weight",
    "weight_amax": "weight",
    "input_amax": "input",
    "pre_quant_scale": "input",
}


def stage_targets(model: nn.Module, stage: AlgoStage) -> tuple[set[str], set[str]]:
    """``(modules, quantizers)`` a stage may write, after subtracting its exclusions."""
    modules, quantizers = resolve_targets(model, stage.scope, stage.selector)
    for selector, glob in stage.exclude:
        ex_modules, ex_quantizers = resolve_targets(model, glob, selector)
        quantizers -= ex_quantizers
        if selector == "module_name":
            modules -= ex_modules
    if stage.exclude:
        owned = _index_model(model).quantizers_of
        modules = {m for m in modules if quantizers.intersection(owned.get(m, ()))}
    return modules, quantizers


def role_quantizers(model: nn.Module, stage: AlgoStage) -> dict[str, set[str]]:
    """The stage's in-scope quantizers, split by role."""
    _, quantizers = stage_targets(model, stage)
    weight = {q for q in quantizers if "weight_quantizer" in q}
    return {"weight": weight, "input": quantizers - weight}


def effective_produces(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Tokens a stage actually writes here — declared ``produces`` minus roles it cannot reach."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.produces if by_role[TOKEN_ROLE.get(t, "weight")]}


def effective_consumes(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Non-ambient tokens a stage reads here."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.consumes - ALWAYS_AVAILABLE if by_role[TOKEN_ROLE.get(t, "weight")]}


def token_targets(model: nn.Module, stage: AlgoStage, token: str) -> set[str]:
    """The quantizers on which ``stage`` reads or writes ``token``."""
    return role_quantizers(model, stage)[TOKEN_ROLE.get(token, "weight")]


def _token_overlap(model: nn.Module, a: AlgoStage, b: AlgoStage, token: str) -> bool:
    return bool(token_targets(model, a, token) & token_targets(model, b, token))


def stage_predicate(model: nn.Module, stage: AlgoStage) -> Callable[[nn.Module], bool]:
    """Build the ``should_process`` write-mask for a stage."""
    modules, quantizers = stage_targets(model, stage)
    allowed = {id(model.get_submodule(name)) for name in modules | quantizers}
    return lambda module: id(module) in allowed


# --- Lowering ---


def _algo_to_name_and_cfg(algo) -> tuple[str | None, dict]:
    """Normalize one pipeline element to ``(algo_name, kwargs)``."""
    if isinstance(algo, QuantizeAlgorithmConfig):
        algo = algo.model_dump()
    if algo is None or isinstance(algo, str):
        return algo, {"method": algo}
    if isinstance(algo, dict):
        if "method" not in algo:
            raise AlgoCfgValidationError(
                f"Algorithm dict must have a 'method' key; got {sorted(algo)}. Entry: {algo!r}"
            )
        return algo["method"], dict(algo)
    raise AlgoCfgValidationError(f"Invalid algorithm config type {type(algo)}: {algo!r}")


def _lower(entries: Iterable[AlgoCfgEntry], algorithm) -> CalibrationPlan:
    """Config -> stages.  No model needed; validation of names happens separately."""
    plan: CalibrationPlan = []
    for e_idx, entry in enumerate(entries):
        selector, scope = entry.selector
        for order, algo in enumerate(entry.cfg):
            name, cfg = _algo_to_name_and_cfg(algo)
            plan.append(AlgoStage(name, cfg, scope, selector, order, e_idx))

    # `algorithm` is the same thing at scope "*", as a fallback: it must not re-run over
    # targets an entry already claimed.
    if algorithm is not None:
        claimed = tuple(entry.selector for entry in entries)
        algos = algorithm if isinstance(algorithm, list) else [algorithm]
        for order, algo in enumerate(algos):
            name, cfg = _algo_to_name_and_cfg(algo)
            if name is None:
                continue
            plan.append(AlgoStage(name, cfg, "*", "quantizer_name", order, -1, exclude=claimed))
    return plan


# --- Validation ---

#: Linears fused into one kernel at export: they share a weight scale, so also a pipeline.
FUSED_SIBLING_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("w1", "w3"),
)


def _report(msg: str, strict: bool = True, sink: list[str] | None = None) -> None:
    if sink is not None:
        sink.append(msg)
        return
    if strict:
        raise AlgoCfgValidationError(msg)
    warnings.warn(f"algo_cfg: {msg}", stacklevel=3)


def known_algorithms() -> list[str]:
    """Algorithm names currently registered in the calibrate-mode registry."""
    from .mode import CalibrateModeRegistry

    names = getattr(CalibrateModeRegistry, "_name2descriptor", {})
    return sorted(
        n.removesuffix("_calibrate")
        for n in names
        if n.endswith("_calibrate") and not n.startswith("_")
    )


def _validate_config_only(plan: CalibrationPlan) -> None:
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    for stage in plan:
        mode_name = BaseCalibrateModeDescriptor._get_mode_name(stage.algo)
        if mode_name not in CalibrateModeRegistry:
            raise AlgoCfgValidationError(
                f"unknown algorithm {stage.algo!r}. Known algorithms: {known_algorithms()}"
            )


def _validate_scopes(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    for stage in plan:
        modules, quantizers = resolve_targets(model, stage.scope, stage.selector)
        if not modules and not quantizers:
            _report(
                f"scope {stage.selector}={stage.scope!r} (stage {stage}) matches no target in "
                "the model. Check the glob against the quantized module/quantizer names.",
                sink=sink,
            )
            continue

        caps = stage.capabilities
        if caps is None:
            continue

        if not caps.honors_write_mask:
            everything = set(_index_model(model).quantizers)
            _, in_scope = stage_targets(model, stage)
            if in_scope != everything:
                _report(
                    f"{stage.algo!r} does not honour the scoping write-mask, so it cannot be "
                    f"restricted to {stage.selector}={stage.scope!r} ({len(in_scope)} of "
                    f"{len(everything)} quantizers) -- it would write outside its scope and "
                    "clobber other stages. Use it at whole-model scope, or add "
                    "`should_process` support to its calibration function first.",
                    sink=sink,
                )
                continue
        # A whole-module algorithm writes every quantizer of its linears, so its scope must be
        # closed under module ownership -- otherwise it writes outside the mask and
        # `effective_produces` understates it, hiding real conflicts.
        if caps.writes_whole_module:
            modules, quantizers = stage_targets(model, stage)
            owned = _index_model(model).quantizers_of
            unreachable = {q for m in modules for q in owned.get(m, ()) if q not in quantizers}
            if unreachable:
                _report(
                    f"{stage.algo!r} writes whole modules: it touches every quantizer of "
                    f"the modules it touches, but {stage.selector}={stage.scope!r} leaves "
                    f"{len(unreachable)} of them out of scope (e.g. "
                    f"{sorted(unreachable)[0]!r}). It would write them anyway, outside the "
                    "write-mask. Select the modules instead, with `module_name`.",
                    sink=sink,
                )
                continue

        if stage.selector == "quantizer_name":
            roles = {"weight" if "weight_quantizer" in q else "input" for q in quantizers}
            if caps.optimizes != "both" and roles and caps.optimizes not in roles:
                _report(
                    f"{stage.algo!r} only improves {caps.optimizes} quantizers but "
                    f"{stage.selector}={stage.scope!r} matches only {sorted(roles)} quantizers "
                    "— the stage would be a no-op.",
                    sink=sink,
                )


def _validate_fused_siblings(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    pipeline_of: dict[str, tuple[str | None, ...]] = {}
    for stage in plan:
        modules, _ = stage_targets(model, stage)
        for m in modules:
            pipeline_of[m] = (*pipeline_of.get(m, ()), stage.algo)

    for group in FUSED_SIBLING_GROUPS:
        by_parent: dict[str, dict[str, tuple]] = {}
        for linear in _index_model(model).linears:
            leaf = linear.rsplit(".", 1)[-1]
            if leaf in group:
                by_parent.setdefault(linear.rsplit(".", 1)[0], {})[leaf] = pipeline_of.get(
                    linear, ()
                )
        for parent, members in by_parent.items():
            if len(set(members.values())) > 1:
                _report(
                    f"fusible siblings under {parent!r} got different pipelines "
                    f"({ {k: list(v) for k, v in members.items()} }). They export to one fused "
                    "kernel and must share a single weight scale, so they must share one "
                    "pipeline.",
                    sink=sink,
                )


def _validate_dependencies(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    for i, stage in enumerate(plan):
        caps = stage.capabilities
        if caps is None:
            continue

        # (1) Non-composable repeat: an earlier stage produced a token this algorithm needs
        #     absent to be correct.
        for j in range(i):
            prev = plan[j]
            if prev.capabilities is None:
                continue
            clash = {
                t
                for t in caps.conflicts_with & effective_produces(model, prev)
                if _token_overlap(model, stage, prev, t)
            }
            if clash:
                _report(
                    f"stage {i} ({stage}) cannot follow stage {j} ({prev}) on overlapping "
                    f"targets: {stage.algo!r} assumes {sorted(clash)} is not already set, but "
                    f"{prev.algo!r} produces it. Re-running it folds the scale a second time "
                    "while keeping only the last activation-side scale. Insert an explicit "
                    "unfold (disable_pre_quant_scale_and_resmooth) between them, or drop the "
                    "repeat.",
                    sink=sink,
                )

        # (2) Dead stage: everything it writes is overwritten before being read.
        produced = effective_produces(model, stage)
        if not produced:
            continue
        overwriters: dict[str, AlgoStage] = {}
        for token in produced:
            for j in range(i + 1, len(plan)):
                later = plan[j]
                if later.capabilities is None or not _token_overlap(model, stage, later, token):
                    continue
                if token in effective_consumes(model, later):
                    break  # somebody read it -- not dead
                if token in effective_produces(model, later):
                    overwriters[token] = later
                    break
        if set(overwriters) == produced:
            first = next(iter(overwriters.values()))
            _report(
                f"stage {i} ({stage}) is dead: everything it produces ({sorted(produced)}) is "
                f"overwritten by a later stage ({first}) on the same quantizers, without being "
                "read in between. Remove it, or move it after the stage that overwrites it.",
                sink=sink,
            )


# --- Entry point ---


def compile_algo_cfg(
    config: QuantizeConfig | dict,
    model: nn.Module | None = None,
    strict: bool = True,
) -> CalibrationPlan:
    """Lower a quantize config into an ordered, validated list of scoped stages."""
    if isinstance(config, QuantizeConfig):
        entries, algorithm = config.algo_cfg or [], config.algorithm
    else:
        raw_entries = config.get("algo_cfg") or []
        entries = [e if isinstance(e, AlgoCfgEntry) else AlgoCfgEntry(**e) for e in raw_entries]
        algorithm = config.get("algorithm", "max")

    # An explicit algo_cfg suppresses the implicit whole-model default; `algorithm` only
    # fills in what entries do not cover.
    if entries and algorithm is not None:
        covered = _coverage_is_total(model, entries) if model is not None else False
        if covered:
            algorithm = None

    plan = _lower(entries, algorithm)
    _validate_config_only(plan)
    if model is not None:
        violations: list[str] = []
        with _reusing_model_index(model):
            _validate_scopes(model, plan, violations)
            _validate_fused_siblings(model, plan, violations)
            _validate_dependencies(model, plan, violations)
        if violations:
            body = "\n".join(f"  {i + 1}. {v}" for i, v in enumerate(violations))
            msg = f"invalid algo_cfg ({len(violations)} problem(s)):\n{body}"
            if strict:
                raise AlgoCfgValidationError(msg)
            warnings.warn(f"algo_cfg: {msg}", stacklevel=2)
    return plan


def _coverage_is_total(model: nn.Module, entries: list[AlgoCfgEntry]) -> bool:
    """Whether the entries already cover every quantizer, making `algorithm` redundant."""
    index = _index_model(model)
    covered: set[str] = set()
    for entry in entries:
        selector, scope = entry.selector
        _, quantizers = resolve_targets(model, scope, selector)
        covered |= quantizers
    return covered >= set(index.quantizers)


#: Tokens that never need producing: the weight is part of the model, acts come from the
#: forward loop.
ALWAYS_AVAILABLE = frozenset({"weight", "acts"})


def derive_handoff(model: nn.Module, plan: CalibrationPlan, i: int) -> dict:
    """Extra kwargs for stage ``i`` implied by what earlier stages already produced."""
    stage = plan[i]
    if stage.capabilities is None:
        return {}
    with _reusing_model_index(model):
        needed = effective_consumes(model, stage)
        if not needed:
            return {}

        # Coverage, not overlap: skipping init is only safe if *every* target already has the
        # state. A narrow producer before a wide consumer would leave some with no amax.
        for token in needed:
            produced_on: set[str] = set()
            for j in range(i):
                if plan[j].capabilities is None:
                    continue
                if token in effective_produces(model, plan[j]):
                    produced_on |= token_targets(model, plan[j], token)
            if not token_targets(model, stage, token) <= produced_on:
                return {}
    return {"skip_max_init": True}


def _stage_targets(model: nn.Module, stage: AlgoStage) -> set[str]:
    modules, quantizers = stage_targets(model, stage)
    return modules | quantizers


def plan_hash(plan: CalibrationPlan) -> str:
    """A stable hash of the plan."""
    import hashlib

    payload = "|".join(str(s.key()) for s in plan)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe_plan(plan: CalibrationPlan, model: nn.Module | None = None) -> str:
    """Human-readable plan dump, used by the demos and for debugging."""
    if not plan:
        return "  (empty plan — no calibration)"
    lines = []
    for i, stage in enumerate(plan):
        suffix = ""
        if model is not None:
            modules, quantizers = stage_targets(model, stage)
            suffix = f"  -> {len(modules)} module(s), {len(quantizers)} quantizer(s)"
        lines.append(f"  [{i}] {stage}{suffix}")
    return "\n".join(lines)
