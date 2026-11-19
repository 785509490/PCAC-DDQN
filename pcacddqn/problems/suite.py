"""The 23-problem roster.

Class meanings (unchanged from the MATLAB version):

    A  calibration      no conditioning should pay
    B  schedule-public  the time table is a published constant, so a lookup is a complete solution
    C  online           the time table is redrawn every run, so only an online method can react
    D  regime-switch    a new motion kind every few environments
    E  random jumps     E1-E3 heterogeneous (stress), E4-E6 homogeneous (negative control)

B1, B2 and B3 are the same problems as C1, C2 and C4 in every respect except
`seedJ`: a fixed published schedule versus a per-run draw. That single line is
the central comparison of the benchmark. Each B problem gets its OWN published
schedule (20250901/02/03): sharing one table would make the B class a single
draw, not a class.
"""

from __future__ import annotations

from .dynamic import DynamicProblem, ProblemSpec

PUBLIC_SCHEDULES = (20250901, 20250902, 20250903)


def _spec(**kw) -> ProblemSpec:
    return ProblemSpec(**kw)


SPECS: list[ProblemSpec] = [
    # -- A: calibration ----------------------------------------------------
    _spec(pid="A1", cls="A", base="quad", mode="static",
          geometry="convex", change="none", law="drift",
          intensity=False,
          note="static problem: conditioning must be zero"),
    _spec(pid="A2", cls="A", base="quad", mode="translate",
          geometry="convex", change="none", law="drift",
          intensity=False,
          note="linear drift, fixed severity: conditioning must be zero"),

    # -- B: public schedule ------------------------------------------------
    _spec(pid="B1", cls="B", base="quad", mode="translate",
          geometry="convex", change="none", law="regime", reg_len=3,
          schedule_seed=PUBLIC_SCHEDULES[0]),
    _spec(pid="B2", cls="B", base="multi", base_amp=0.25, mode="shape",
          geometry="convex", change="none", law="regime", reg_len=3,
          schedule_seed=PUBLIC_SCHEDULES[1]),
    _spec(pid="B3", cls="B", base="rot", mode="translate",
          geometry="convex", change="scale", law="regime", reg_len=3,
          schedule_seed=PUBLIC_SCHEDULES[2]),

    # -- C: online ---------------------------------------------------------
    _spec(pid="C1", cls="C", base="quad", mode="translate",
          geometry="convex", change="none", law="regime", reg_len=3),
    _spec(pid="C2", cls="C", base="multi", base_amp=0.25, mode="shape",
          geometry="convex", change="none", law="regime", reg_len=3),
    _spec(pid="C3", cls="C", base="link", mode="chain",
          geometry="linear", change="none", law="regime", reg_len=3),
    _spec(pid="C4", cls="C", base="rot", mode="translate",
          geometry="convex", change="scale", law="regime", reg_len=3),
    _spec(pid="C5", cls="C", base="quad", mode="shape",
          geometry="concave", change="curvature", law="regime", reg_len=3),
    _spec(pid="C6", cls="C", base="rot", mode="roleswitch",
          geometry="sphere", change="scale", law="regime", reg_len=3, M=3),

    # -- D: regime switching every environment -----------------------------
    _spec(pid="D1", cls="D", base="rot", mode="roleswitch",
          geometry="convex", change="none", law="regime", reg_len=1),
    _spec(pid="D2", cls="D", base="multi", base_amp=0.25, mode="shape",
          geometry="disconnected", change="shift", law="regime", reg_len=1),
    _spec(pid="D3", cls="D", base="link", mode="chain",
          geometry="concave", change="scale", law="regime", reg_len=1),
    _spec(pid="D4", cls="D", base="quad", mode="roleswitch",
          geometry="convex", change="curvature", law="regime", reg_len=1),
    _spec(pid="D5", cls="D", base="rot", mode="chain",
          geometry="linear", change="shift", law="regime", reg_len=1),
    _spec(pid="D6", cls="D", base="multi", base_amp=0.25, mode="translate",
          geometry="sphere", change="curvature", law="regime", reg_len=1, M=3),

    # -- E: random jumps ---------------------------------------------------
    _spec(pid="E1", cls="E", base="quad", mode="translate",
          geometry="convex", change="none", law="jump", jump_variant="hetero"),
    _spec(pid="E2", cls="E", base="multi", base_amp=0.25, mode="shape",
          geometry="convex", change="none", law="jump", jump_variant="hetero"),
    _spec(pid="E3", cls="E", base="link", mode="chain",
          geometry="sphere", change="none", law="jump", jump_variant="hetero",
          M=3),
    _spec(pid="E4", cls="E", base="quad", mode="translate",
          geometry="convex", change="none", law="jump", jump_variant="homo",
          note="negative control: fixed jump magnitude"),
    _spec(pid="E5", cls="E", base="multi", base_amp=0.25, mode="shape",
          geometry="convex", change="none", law="jump", jump_variant="homo",
          note="negative control: fixed jump magnitude"),
    _spec(pid="E6", cls="E", base="link", mode="chain",
          geometry="sphere", change="none", law="jump", jump_variant="homo",
          M=3, note="negative control: fixed jump magnitude"),
]

SPEC_BY_ID = {s.pid: s for s in SPECS}

# -- F: segmented motion ---------------------------------------------------
# Not part of the 23-problem roster. These problems change the *kind* of
# motion mid-run, so the relevance of the archive switches within a single
# run. They exist to test online memory control, not to replace the roster.
SEGMENTED_SPECS: list[ProblemSpec] = [
    _spec(pid="F1", cls="F", base="quad", mode="seg_translate_shape",
          geometry="convex", change="none", law="drift"),
    _spec(pid="F2", cls="F", base="multi", base_amp=0.25,
          mode="seg_shape_translate", geometry="convex", change="none",
          law="drift"),
    _spec(pid="F3", cls="F", base="link", mode="seg_translate_chain",
          geometry="linear", change="none", law="regime", reg_len=3),
    _spec(pid="F4", cls="F", base="multi", base_amp=0.25,
          mode="seg_chain_translate", geometry="sphere",
          change="curvature", law="regime", reg_len=3, M=3),
]
SEGMENTED_BY_ID = {s.pid: s for s in SEGMENTED_SPECS}


def build_segmented(pid: str, instance_seed: int = 0, D: int = 10,
                    n_env: int = 21, taut: int = 20, T0: int = 50,
                    intensity_log_sd: float = 0.45,
                    intensity_rho: float = 0.7,
                    intensity_mode: str = "normal",
                    intensity_seed: str | None = None) -> DynamicProblem:
    pro = DynamicProblem(SEGMENTED_BY_ID[pid], D=D, n_env=n_env, taut=taut,
                         T0=T0, intensity_log_sd=intensity_log_sd,
                         intensity_rho=intensity_rho,
                         intensity_mode=intensity_mode,
                         intensity_seed=intensity_seed)
    pro.reseed(instance_seed)
    return pro


def build(pid: str, instance_seed: int = 0, D: int = 10, n_env: int = 21,
          taut: int = 20, T0: int = 50,
          intensity_log_sd: float = 0.45,
          intensity_rho: float = 0.7,
          intensity_mode: str = "normal",
          intensity_seed: str | None = None) -> DynamicProblem:
    pro = DynamicProblem(SPEC_BY_ID[pid], D=D, n_env=n_env, taut=taut, T0=T0,
                         intensity_log_sd=intensity_log_sd,
                         intensity_rho=intensity_rho,
                         intensity_mode=intensity_mode,
                         intensity_seed=intensity_seed)
    pro.reseed(instance_seed)
    return pro


def suite() -> list[ProblemSpec]:
    return list(SPECS)


def class_of(pid: str) -> str:
    return SPEC_BY_ID[pid].cls


def ids(cls: str | None = None) -> list[str]:
    out = SPECS
    if cls is not None:
        out = [s for s in out if s.cls == cls]
    return [s.pid for s in out]
