from __future__ import annotations

import argparse
import ast
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable, Mapping

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


# =========================
# GUROBI WLS INLINE LICENSE
# =========================
# DÁN THÔNG TIN LICENSE CỦA BẠN VÀO 3 DÒNG BÊN DƯỚI.
# Nếu để trống/0, code sẽ fallback về cách tạo model Gurobi như cũ.
WLS_PARAMS = {
    "WLSACCESSID": "79eae300-197a-4836-9b9b-dcb0e4396a25",
    "WLSSECRET": "2a92e261-8cf9-4b88-8d31-6ea359e4bc3f",
    "LICENSEID": 2806284,
}


def _build_gurobi_env_from_inline_wls() -> Optional["gp.Env"]:
    if gp is None:
        return None

    access_id = str(WLS_PARAMS.get("WLSACCESSID", "")).strip()
    secret = str(WLS_PARAMS.get("WLSSECRET", "")).strip()
    raw_license_id = WLS_PARAMS.get("LICENSEID", 0)
    try:
        license_id = int(raw_license_id)
    except Exception:
        license_id = 0

    if access_id and secret and license_id > 0:
        params = {
            "WLSACCESSID": access_id,
            "WLSSECRET": secret,
            "LICENSEID": license_id,
        }
        return gp.Env(params=params)
    return None


@dataclass
class Depot:
    x: float
    y: float


@dataclass
class CandidatePL:
    id: int
    x: float
    y: float
    fixed_costs: Dict[str, float]


@dataclass
class ServiceRegion:
    id: int
    x: float
    y: float
    demand: Dict[int, float]


@dataclass
class PLLRPInstance:
    periods: List[int]
    modules: Dict[str, float]
    vehicle_capacity: float
    vehicle_fixed_cost: float
    travel_cost: float
    compensation_cost: float
    depot: Depot
    candidate_pls: List[CandidatePL] = field(default_factory=list)
    service_regions: List[ServiceRegion] = field(default_factory=list)
    name: str = "pl_lrp_instance"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PLLRPInstance":
        periods = [int(t) for t in data["periods"]]
        modules = {str(k): float(v) for k, v in data["modules"].items()}
        depot = Depot(**data["depot"])
        candidate_pls = [CandidatePL(
            id=int(item["id"]),
            x=float(item["x"]),
            y=float(item["y"]),
            fixed_costs={str(k): float(v) for k, v in item["fixed_costs"].items()},
        ) for item in data["candidate_pls"]]
        service_regions = [ServiceRegion(
            id=int(item["id"]),
            x=float(item["x"]),
            y=float(item["y"]),
            demand={int(k): float(v) for k, v in item["demand"].items()},
        ) for item in data["service_regions"]]
        return PLLRPInstance(
            periods=periods,
            modules=modules,
            vehicle_capacity=float(data["vehicle_capacity"]),
            vehicle_fixed_cost=float(data["vehicle_fixed_cost"]),
            travel_cost=float(data["travel_cost"]),
            compensation_cost=float(data["compensation_cost"]),
            depot=depot,
            candidate_pls=candidate_pls,
            service_regions=service_regions,
            name=str(data.get("name", "pl_lrp_instance")),
        )

    @staticmethod
    def from_json(path: str | Path) -> "PLLRPInstance":
        with open(path, "r", encoding="utf-8") as f:
            return PLLRPInstance.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "periods": self.periods,
            "modules": self.modules,
            "vehicle_capacity": self.vehicle_capacity,
            "vehicle_fixed_cost": self.vehicle_fixed_cost,
            "travel_cost": self.travel_cost,
            "compensation_cost": self.compensation_cost,
            "depot": {"x": self.depot.x, "y": self.depot.y},
            "candidate_pls": [
                {
                    "id": pl.id,
                    "x": pl.x,
                    "y": pl.y,
                    "fixed_costs": pl.fixed_costs,
                }
                for pl in self.candidate_pls
            ],
            "service_regions": [
                {
                    "id": sr.id,
                    "x": sr.x,
                    "y": sr.y,
                    "demand": sr.demand,
                }
                for sr in self.service_regions
            ],
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _maybe_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == "":
        return value
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except Exception:
        return value


def _parse_start_key(raw_key: Any, expected_len: int) -> Optional[Any]:
    if expected_len == 1 and not isinstance(raw_key, (tuple, list)):
        return _maybe_number(raw_key)

    seq: Optional[List[Any]] = None
    if isinstance(raw_key, tuple):
        seq = list(raw_key)
    elif isinstance(raw_key, list):
        seq = list(raw_key)
    elif isinstance(raw_key, str):
        s = raw_key.strip()
        if s in {"", "u"} and expected_len == 1:
            return s
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (tuple, list)):
                seq = list(parsed)
            elif expected_len == 1:
                return _maybe_number(parsed)
        except Exception:
            if "," in s:
                seq = [part.strip() for part in s.split(",")]
            elif expected_len == 1:
                return _maybe_number(s)
    if seq is None or len(seq) != expected_len:
        return None
    seq = [_maybe_number(x) for x in seq]
    return tuple(seq)


def _load_section_as_map(section: Any, key_len: int) -> Dict[Any, float]:
    result: Dict[Any, float] = {}
    if section is None:
        return result

    if isinstance(section, Mapping):
        for k, v in section.items():
            parsed_key = _parse_start_key(k, key_len)
            if parsed_key is None:
                if key_len == 1 and isinstance(v, Mapping) and k == "u" and "u" in v:
                    result["u"] = float(v["u"])
                continue
            result[parsed_key] = float(v)
        return result

    if isinstance(section, list):
        for item in section:
            if isinstance(item, Mapping):
                if "key" in item and "value" in item:
                    parsed_key = _parse_start_key(item["key"], key_len)
                    if parsed_key is not None:
                        result[parsed_key] = float(item["value"])
                    continue
                if "indices" in item and "value" in item:
                    parsed_key = _parse_start_key(item["indices"], key_len)
                    if parsed_key is not None:
                        result[parsed_key] = float(item["value"])
                    continue
                if "index" in item and "value" in item:
                    parsed_key = _parse_start_key(item["index"], key_len)
                    if parsed_key is not None:
                        result[parsed_key] = float(item["value"])
                    continue
            if isinstance(item, (tuple, list)) and len(item) == key_len + 1:
                parsed_key = _parse_start_key(item[:key_len], key_len)
                if parsed_key is not None:
                    result[parsed_key] = float(item[key_len])
        return result

    raise ValueError(f"Unsupported warm-start section format: {type(section)!r}")


def _key_aliases(key: Any) -> Iterable[Any]:
    if isinstance(key, tuple):
        aliases = set()
        choices: List[List[Any]] = []
        for elem in key:
            opts = [elem]
            if isinstance(elem, str):
                maybe_num = _maybe_number(elem)
                if maybe_num != elem:
                    opts.append(maybe_num)
            elif isinstance(elem, (int, float)):
                opts.append(str(int(elem) if isinstance(elem, float) and elem.is_integer() else elem))
            choices.append(opts)
        def _backtrack(idx: int, cur: List[Any]) -> None:
            if idx == len(choices):
                aliases.add(tuple(cur))
                return
            for val in choices[idx]:
                cur.append(val)
                _backtrack(idx + 1, cur)
                cur.pop()
        _backtrack(0, [])
        return aliases
    aliases = {key}
    if isinstance(key, str):
        aliases.add(_maybe_number(key))
    elif isinstance(key, (int, float)):
        aliases.add(str(int(key) if isinstance(key, float) and key.is_integer() else key))
    return aliases


def _remap_start_keys_to_model(section_map: Dict[Any, float], model_keys: Iterable[Any]) -> Dict[Any, float]:
    alias_to_actual: Dict[Any, Any] = {}
    for actual_key in model_keys:
        for alias in _key_aliases(actual_key):
            alias_to_actual[alias] = actual_key
    remapped: Dict[Any, float] = {}
    for raw_key, value in section_map.items():
        actual = None
        for alias in _key_aliases(raw_key):
            if alias in alias_to_actual:
                actual = alias_to_actual[alias]
                break
        if actual is not None:
            remapped[actual] = value
    return remapped



def load_warm_start_payload(path: str | Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError("Warm-start JSON must be a mapping at the top level.")
    return data


def validate_warm_start_metadata(meta: Optional[Mapping[str, Any]], instance: PLLRPInstance) -> None:
    if not meta:
        return

    def _norm_list(values: Any) -> List[int]:
        return [int(v) for v in values]

    expected_module_ids = sorted(int(k) for k in instance.modules.keys())
    expected_pl_ids = sorted(pl.id for pl in instance.candidate_pls)
    expected_sr_ids = sorted(sr.id for sr in instance.service_regions)

    checks = []
    if meta.get("instance_name") is not None:
        checks.append(("instance_name", str(instance.name), str(meta.get("instance_name"))))
    if meta.get("periods") is not None:
        checks.append(("periods", list(instance.periods), _norm_list(meta["periods"])))
    if meta.get("module_ids") is not None:
        checks.append(("module_ids", expected_module_ids, sorted(_norm_list(meta["module_ids"]))))
    if meta.get("pl_ids") is not None:
        checks.append(("pl_ids", expected_pl_ids, sorted(_norm_list(meta["pl_ids"]))))
    if meta.get("sr_ids") is not None:
        checks.append(("sr_ids", expected_sr_ids, sorted(_norm_list(meta["sr_ids"]))))

    mismatches = [
        f"{name}: expected={expected}, warm_start={actual}"
        for name, expected, actual in checks
        if expected != actual
    ]
    if mismatches:
        raise ValueError("Warm-start metadata does not match instance. " + " | ".join(mismatches))


def load_warm_start_file(path: str | Path, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    data = payload if payload is not None else load_warm_start_payload(path)
    starts_root = data.get("starts", data) if isinstance(data, Mapping) else data
    if not isinstance(starts_root, Mapping):
        raise ValueError("Warm-start JSON must be a mapping or contain a top-level 'starts' mapping.")

    return {
        "z": _load_section_as_map(starts_root.get("z"), 2),
        "y": _load_section_as_map(starts_root.get("y"), 2),
        "v": _load_section_as_map(starts_root.get("v"), 2),
        "q": _load_section_as_map(starts_root.get("q"), 2),
        "I": _load_section_as_map(starts_root.get("I"), 2),
        "x": _load_section_as_map(starts_root.get("x"), 3),
        "w": _load_section_as_map(starts_root.get("w"), 2),
        "u": _load_section_as_map(starts_root.get("u"), 1),
    }


def apply_warm_start_to_model(model: gp.Model, vars_dict: Dict[str, Any], warm_data: Dict[str, Any]) -> Dict[str, int]:
    applied_counts: Dict[str, int] = {}
    for name in ["z", "y", "v", "q", "I", "x", "w"]:
        var_container = vars_dict[name]
        remapped = _remap_start_keys_to_model(warm_data.get(name, {}), var_container.keys())
        applied_counts[name] = len(remapped)
        for key, value in remapped.items():
            var_container[key].Start = value

    u_map = warm_data.get("u", {})
    u_start = None
    if "u" in u_map:
        u_start = u_map["u"]
    elif 0 in u_map:
        u_start = u_map[0]
    elif "0" in u_map:
        u_start = u_map["0"]
    if u_start is not None:
        vars_dict["u"].Start = u_start
        applied_counts["u"] = 1
    else:
        applied_counts["u"] = 0

    model.update()
    return applied_counts


def generate_random_instance(
    n_pl: int,
    n_sr: int,
    n_modules: int,
    n_periods: int,
    demand_pattern: str = "uniform",
    seed: int = 0,
    side_length: float = 25.0,
    name: Optional[str] = None,
) -> PLLRPInstance:
    rng = random.Random(seed)
    periods = list(range(1, n_periods + 1))

    modules = {str(m): float(50 * m) for m in range(1, n_modules + 1)}

    def random_pl_cost(module_index: int) -> float:
        low = 15 * (1 + 2 * module_index)
        high = 20 * (1 + 2 * module_index)
        return 1000.0 * rng.uniform(low, high)

    candidate_pls: List[CandidatePL] = []
    for i in range(1, n_pl + 1):
        fixed_costs = {str(m): random_pl_cost(m) for m in range(1, n_modules + 1)}
        candidate_pls.append(
            CandidatePL(
                id=i,
                x=rng.uniform(0.0, side_length),
                y=rng.uniform(0.0, side_length),
                fixed_costs=fixed_costs,
            )
        )

    def bimodal_expected(period_idx: int, total: int) -> float:
        # Peaks at both ends, valley in the middle. Average remains around 5.
        if total == 1:
            return 5.0
        center = (total - 1) / 2.0
        dist = abs(period_idx - center)
        max_dist = max(center, 1e-6)
        base = 2.5 + 5.0 * (dist / max_dist)
        return min(max(base, 0.0), 10.0)

    service_regions: List[ServiceRegion] = []
    for k in range(1, n_sr + 1):
        demand: Dict[int, float] = {}
        for pos, t in enumerate(periods):
            if demand_pattern.lower() == "uniform":
                demand[t] = float(rng.randint(0, 10))
            elif demand_pattern.lower() == "bimodal":
                mean_val = bimodal_expected(pos, n_periods)
                val = int(round(max(0.0, min(10.0, rng.gauss(mean_val, 1.5)))))
                demand[t] = float(max(0, min(10, val)))
            else:
                raise ValueError("demand_pattern must be 'uniform' or 'bimodal'")
        service_regions.append(
            ServiceRegion(
                id=k,
                x=rng.uniform(0.0, side_length),
                y=rng.uniform(0.0, side_length),
                demand=demand,
            )
        )

    return PLLRPInstance(
        periods=periods,
        modules=modules,
        vehicle_capacity=400.0,
        vehicle_fixed_cost=50_000.0,
        travel_cost=1_370.0,
        compensation_cost=110.0,
        depot=Depot(side_length / 2.0, side_length / 2.0),
        candidate_pls=candidate_pls,
        service_regions=service_regions,
        name=name or f"pl_lrp_{n_pl}_{n_sr}_{n_modules}_{n_periods}_{demand_pattern}_{seed}",
    )


class PLLRPBranchAndCut:
    """
    Practical branch-and-cut baseline for the PL-LRP in Zadeh et al. (2026).

    Notes:
    - Uses the paper's main MILP (1)-(15) and valid inequalities (16)-(21).
    - Solves with Gurobi's built-in MIP branch-and-cut.
    - The paper's load constraints are somewhat terse for direct implementation; this code
      uses a practical strengthened interpretation of constraints (11)-(12):
        * w is defined only for parcel locker nodes,
        * w[i,t] >= q[i,t] is added as a natural strengthening,
        * load propagation is enforced only between PL-to-PL arcs.
      This preserves the intended route-load logic and subtour elimination behavior for the
      B&C baseline while remaining implementable and numerically stable.
    - This file intentionally implements the B&C baseline only, not the full VMND/LFRS stack.
    """

    def __init__(
        self,
        instance: PLLRPInstance,
        add_valid_inequalities: bool = True,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        log_to_console: bool = True,
        threads: Optional[int] = None,
        numeric_focus: Optional[int] = 1,
        warm_start_path: Optional[str | Path] = None,
    ) -> None:
        if gp is None or GRB is None:
            raise RuntimeError(
                "gurobipy is not available. Install Gurobi and gurobipy to run this model."
            )
        self.instance = instance
        self.add_valid_inequalities = add_valid_inequalities
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.log_to_console = log_to_console
        self.threads = threads
        self.numeric_focus = numeric_focus
        self.warm_start_path = Path(warm_start_path) if warm_start_path is not None else None
        self.warm_start_applied: bool = False
        self.warm_start_counts: Dict[str, int] = {}
        self.warm_start_payload: Optional[Mapping[str, Any]] = None
        self.warm_start_summary: Dict[str, Any] = {}
        self.env: Optional["gp.Env"] = None

        self.model: Optional[gp.Model] = None
        self.vars: Dict[str, Any] = {}
        self.data: Dict[str, Any] = {}

    def _prepare_data(self) -> Dict[str, Any]:
        inst = self.instance
        periods = list(inst.periods)
        period_pos = {t: idx for idx, t in enumerate(periods)}
        first_period = periods[0]

        pl_ids = [pl.id for pl in inst.candidate_pls]
        sr_ids = [sr.id for sr in inst.service_regions]
        module_ids = list(inst.modules.keys())
        nodes0prime = [0] + pl_ids
        arcs = [(i, j, t) for t in periods for i in nodes0prime for j in nodes0prime if i != j]
        pl_to_pl_arcs = [(i, j, t) for t in periods for i in pl_ids for j in pl_ids if i != j]

        pl_by_id = {pl.id: pl for pl in inst.candidate_pls}
        sr_by_id = {sr.id: sr for sr in inst.service_regions}

        dist_nodes = {}
        for i in nodes0prime:
            xi, yi = (inst.depot.x, inst.depot.y) if i == 0 else (pl_by_id[i].x, pl_by_id[i].y)
            for j in nodes0prime:
                if i == j:
                    continue
                xj, yj = (inst.depot.x, inst.depot.y) if j == 0 else (pl_by_id[j].x, pl_by_id[j].y)
                dist_nodes[(i, j)] = euclidean(xi, yi, xj, yj)

        dist_assign = {}
        demand = {}
        for i in pl_ids:
            xi, yi = pl_by_id[i].x, pl_by_id[i].y
            for k in sr_ids:
                xk, yk = sr_by_id[k].x, sr_by_id[k].y
                dist_assign[(i, k)] = euclidean(xi, yi, xk, yk)
        for sr in inst.service_regions:
            for t in periods:
                demand[(sr.id, t)] = float(sr.demand.get(t, 0.0))

        max_module_capacity = max(inst.modules.values())
        mtilde = {}
        for t in periods:
            remaining = sum(demand[(k, tau)] for k in sr_ids for tau in periods if period_pos[tau] >= period_pos[t])
            mtilde[t] = min(inst.vehicle_capacity, max_module_capacity, remaining)

        d_min = min((demand[(k, t)] for k in sr_ids for t in periods), default=0.0)
        eps = 1e-6 if abs(d_min) < 1e-9 else 0.0
        lambda_first = {k: 1 if demand[(k, first_period)] > 0 else 0 for k in sr_ids}

        fixed_costs = {(pl.id, m): float(pl.fixed_costs[m]) for pl in inst.candidate_pls for m in module_ids}
        module_cap = {m: float(inst.modules[m]) for m in module_ids}

        return {
            "periods": periods,
            "first_period": first_period,
            "period_pos": period_pos,
            "pl_ids": pl_ids,
            "sr_ids": sr_ids,
            "module_ids": module_ids,
            "nodes0prime": nodes0prime,
            "arcs": arcs,
            "pl_to_pl_arcs": pl_to_pl_arcs,
            "dist_nodes": dist_nodes,
            "dist_assign": dist_assign,
            "demand": demand,
            "mtilde": mtilde,
            "d_min": d_min,
            "eps": eps,
            "lambda_first": lambda_first,
            "fixed_costs": fixed_costs,
            "module_cap": module_cap,
        }

    def build_model(self) -> gp.Model:
        d = self._prepare_data()
        self.data = d
        inst = self.instance

        self.env = _build_gurobi_env_from_inline_wls()
        model = gp.Model(inst.name, env=self.env) if self.env is not None else gp.Model(inst.name)
        model.Params.OutputFlag = 1 if self.log_to_console else 0
        if self.time_limit is not None:
            model.Params.TimeLimit = float(self.time_limit)
        if self.mip_gap is not None:
            model.Params.MIPGap = float(self.mip_gap)
        if self.threads is not None:
            model.Params.Threads = int(self.threads)
        if self.numeric_focus is not None:
            model.Params.NumericFocus = int(self.numeric_focus)

        z = model.addVars(d["pl_ids"], d["module_ids"], vtype=GRB.BINARY, name="z")
        u = model.addVar(vtype=GRB.INTEGER, lb=0.0, name="u")
        y = model.addVars(d["pl_ids"], d["sr_ids"], vtype=GRB.BINARY, name="y")
        x = model.addVars(d["arcs"], vtype=GRB.BINARY, name="x")
        v = model.addVars(d["pl_ids"], d["periods"], vtype=GRB.BINARY, name="v")
        q = model.addVars(d["pl_ids"], d["periods"], lb=0.0, vtype=GRB.CONTINUOUS, name="q")
        w = model.addVars(d["pl_ids"], d["periods"], lb=0.0, vtype=GRB.CONTINUOUS, name="w")
        I = model.addVars(d["pl_ids"], d["periods"], lb=0.0, vtype=GRB.CONTINUOUS, name="I")

        # Objective (1)
        fixed_pl_cost = gp.quicksum(d["fixed_costs"][(i, m)] * z[i, m] for i in d["pl_ids"] for m in d["module_ids"])
        fleet_cost = inst.vehicle_fixed_cost * u
        travel_cost = gp.quicksum(inst.travel_cost * d["dist_nodes"][(i, j)] * x[i, j, t] for (i, j, t) in d["arcs"])
        compensation_cost = gp.quicksum(
            inst.compensation_cost * d["dist_assign"][(i, k)] * d["demand"][(k, t)] * y[i, k]
            for i in d["pl_ids"] for k in d["sr_ids"] for t in d["periods"]
        )
        model.setObjective(fixed_pl_cost + fleet_cost + travel_cost + compensation_cost, GRB.MINIMIZE)

        # (2)
        model.addConstrs(
            (gp.quicksum(y[i, k] for i in d["pl_ids"]) == 1 for k in d["sr_ids"]),
            name="assign_each_sr",
        )

        # (3)
        model.addConstrs(
            (gp.quicksum(z[i, m] for m in d["module_ids"]) <= 1 for i in d["pl_ids"]),
            name="one_module_per_pl",
        )

        # (4), (5), (6), (7)
        for i in d["pl_ids"]:
            cap_expr = gp.quicksum(d["module_cap"][m] * z[i, m] for m in d["module_ids"])
            for pos, t in enumerate(d["periods"]):
                prev_inv = 0.0 if pos == 0 else I[i, d["periods"][pos - 1]]
                demand_alloc = gp.quicksum(d["demand"][(k, t)] * y[i, k] for k in d["sr_ids"])
                model.addConstr(prev_inv + q[i, t] == demand_alloc + I[i, t], name=f"inv_bal_{i}_{t}")
                model.addConstr(demand_alloc + I[i, t] <= cap_expr, name=f"pl_cap_{i}_{t}")
                model.addConstr(v[i, t] <= gp.quicksum(z[i, m] for m in d["module_ids"]), name=f"visit_only_if_open_{i}_{t}")
                model.addConstr(q[i, t] <= d["mtilde"][t] * v[i, t], name=f"qty_only_if_visit_{i}_{t}")

        # (8)
        model.addConstrs(
            (gp.quicksum(x[0, j, t] for j in d["pl_ids"]) <= u for t in d["periods"]),
            name="departures_le_fleet",
        )

        # (9), (10)
        for i in d["pl_ids"]:
            for t in d["periods"]:
                out_expr = gp.quicksum(x[i, j, t] for j in d["nodes0prime"] if j != i)
                in_expr = gp.quicksum(x[j, i, t] for j in d["nodes0prime"] if j != i)
                model.addConstr(out_expr == in_expr, name=f"flow_balance_{i}_{t}")
                model.addConstr(out_expr == v[i, t], name=f"one_vehicle_per_pl_period_{i}_{t}")

        # Strengthened interpretation of (11) and (12)
        for i in d["pl_ids"]:
            for t in d["periods"]:
                model.addConstr(w[i, t] >= q[i, t], name=f"load_ge_delivery_{i}_{t}")
                model.addConstr(w[i, t] <= inst.vehicle_capacity * v[i, t], name=f"load_le_vehiclecap_{i}_{t}")
        for (i, j, t) in d["pl_to_pl_arcs"]:
            model.addConstr(
                w[i, t] - w[j, t] >= q[i, t] - d["mtilde"][t] * (1 - x[i, j, t]),
                name=f"load_prop_{i}_{j}_{t}",
            )

        # (15): I[i,0] = 0 is implicit via first-period balance with prev_inv = 0.

        if self.add_valid_inequalities:
            # (16)
            for t in d["periods"]:
                lhs = gp.quicksum(d["module_cap"][m] * z[i, m] for i in d["pl_ids"] for m in d["module_ids"])
                rhs = gp.quicksum(d["demand"][(k, t)] for k in d["sr_ids"]) + gp.quicksum(I[i, t] for i in d["pl_ids"])
                model.addConstr(lhs >= rhs, name=f"vi_capacity_{t}")

            # (17)
            bound_by_module = {
                m: min(len(d["sr_ids"]), int(math.floor(d["module_cap"][m] / (d["d_min"] + d["eps"]))))
                for m in d["module_ids"]
            }
            for i in d["pl_ids"]:
                model.addConstr(
                    gp.quicksum(y[i, k] for k in d["sr_ids"]) <= gp.quicksum(bound_by_module[m] * z[i, m] for m in d["module_ids"]),
                    name=f"vi_assign_card_{i}",
                )

            # (18)
            for i in d["pl_ids"]:
                open_expr = gp.quicksum(z[i, m] for m in d["module_ids"])
                for k in d["sr_ids"]:
                    model.addConstr(y[i, k] <= open_expr, name=f"vi_open_assign_{i}_{k}")

            # (19)
            ordered_periods = d["periods"]
            for idx, t in enumerate(ordered_periods):
                cumulative_departures = gp.quicksum(x[0, j, tau] for tau in ordered_periods[: idx + 1] for j in d["pl_ids"])
                cumulative_demand = sum(d["demand"][(k, tau)] for k in d["sr_ids"] for tau in ordered_periods[: idx + 1])
                model.addConstr(cumulative_departures >= math.ceil(cumulative_demand / inst.vehicle_capacity), name=f"vi_cum_depart_{t}")

            # (20)
            for idx, t in enumerate(ordered_periods):
                cumulative_deliveries = gp.quicksum(q[i, tau] for i in d["pl_ids"] for tau in ordered_periods[: idx + 1])
                model.addConstr(cumulative_deliveries <= (idx + 1) * inst.vehicle_capacity * u, name=f"vi_cum_ship_{t}")

            # (21)
            first = d["first_period"]
            for i in d["pl_ids"]:
                for k in d["sr_ids"]:
                    if d["lambda_first"][k] == 1:
                        model.addConstr(v[i, first] >= y[i, k], name=f"vi_first_period_{i}_{k}")

        model.update()
        self.model = model
        self.vars = {"z": z, "u": u, "y": y, "x": x, "v": v, "q": q, "w": w, "I": I}

        if self.warm_start_path is not None:
            warm_payload = load_warm_start_payload(self.warm_start_path)
            self.warm_start_payload = warm_payload
            validate_warm_start_metadata(warm_payload.get("meta"), self.instance)
            warm_data = load_warm_start_file(self.warm_start_path, payload=warm_payload)
            self.warm_start_counts = apply_warm_start_to_model(model, self.vars, warm_data)
            self.warm_start_applied = any(count > 0 for count in self.warm_start_counts.values())

        return model

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build_model()
        assert self.model is not None
        self.model.optimize()
        return self.extract_solution()

    def _route_reconstruction(self, x_vals: Dict[Tuple[int, int, int], float]) -> Dict[int, List[List[int]]]:
        periods = self.data["periods"]
        pl_ids = set(self.data["pl_ids"])
        routes: Dict[int, List[List[int]]] = {t: [] for t in periods}
        for t in periods:
            next_from = {(i, j): val for (i, j, tt), val in x_vals.items() if tt == t and val > 0.5}
            starts = [j for (i, j), val in next_from.items() if i == 0 and val > 0.5]
            used_starts = set()
            for start in starts:
                if start in used_starts:
                    continue
                route = [0, start]
                used_starts.add(start)
                cur = start
                visited = {start}
                while True:
                    nxt = None
                    for cand in [n for n in list(pl_ids) + [0] if n != cur]:
                        if (cur, cand) in next_from and next_from[(cur, cand)] > 0.5:
                            nxt = cand
                            break
                    if nxt is None:
                        break
                    route.append(nxt)
                    if nxt == 0:
                        break
                    if nxt in visited:
                        break
                    visited.add(nxt)
                    cur = nxt
                routes[t].append(route)
        return routes


    def _build_warm_start_summary(self, bc_runtime_seconds: Optional[float] = None) -> Dict[str, Any]:
        payload = self.warm_start_payload if isinstance(self.warm_start_payload, Mapping) else {}
        meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
        diagnostics = payload.get("diagnostics", {}) if isinstance(payload, Mapping) else {}
        result_summary = payload.get("result_summary", {}) if isinstance(payload, Mapping) else {}

        lfrs_runtime = None
        for key in ["total_runtime_seconds", "runtime_seconds", "reduced_runtime_seconds"]:
            if isinstance(result_summary, Mapping) and result_summary.get(key) is not None:
                lfrs_runtime = float(result_summary[key])
                break
        if lfrs_runtime is None and isinstance(diagnostics, Mapping):
            for key in ["total_runtime_seconds", "runtime_seconds", "reduced_runtime_seconds"]:
                if diagnostics.get(key) is not None:
                    lfrs_runtime = float(diagnostics[key])
                    break

        lfrs_cost = None
        if isinstance(result_summary, Mapping):
            for key in ["final_total_cost", "total_cost", "objective"]:
                if result_summary.get(key) is not None:
                    lfrs_cost = float(result_summary[key])
                    break
        if lfrs_cost is None and isinstance(diagnostics, Mapping) and diagnostics.get("reduced_obj") is not None:
            lfrs_cost = float(diagnostics["reduced_obj"])

        total_method_runtime = None
        if bc_runtime_seconds is not None and lfrs_runtime is not None:
            total_method_runtime = float(bc_runtime_seconds) + float(lfrs_runtime)

        return {
            "warm_start_path": str(self.warm_start_path) if self.warm_start_path is not None else None,
            "warm_start_used": self.warm_start_applied,
            "warm_start_counts": self.warm_start_counts,
            "meta": meta if isinstance(meta, Mapping) else {},
            "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else {},
            "result_summary": result_summary if isinstance(result_summary, Mapping) else {},
            "lfrs_runtime_seconds": lfrs_runtime,
            "lfrs_cost": lfrs_cost,
            "bc_runtime_seconds": float(bc_runtime_seconds) if bc_runtime_seconds is not None else None,
            "total_method_runtime_seconds": total_method_runtime,
        }

    def extract_solution(self) -> Dict[str, Any]:
        assert self.model is not None
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
        }
        status = status_map.get(self.model.Status, str(self.model.Status))
        sol: Dict[str, Any] = {
            "instance": self.instance.name,
            "status": status,
            "objective": None,
            "best_bound": None,
            "mip_gap": None,
            "runtime_seconds": self.model.Runtime,
            "open_pls": [],
            "fleet_size": None,
            "assignments": {},
            "replenishment": {},
            "inventory": {},
            "routes": {},
            "objective_breakdown": {},
            "warm_start_used": self.warm_start_applied,
            "warm_start_counts": self.warm_start_counts,
            "warm_start_path": str(self.warm_start_path) if self.warm_start_path is not None else None,
            "warm_start_input_summary": self._build_warm_start_summary(self.model.Runtime),
        }
        if self.model.SolCount == 0:
            return sol

        z = self.vars["z"]
        u = self.vars["u"]
        y = self.vars["y"]
        x = self.vars["x"]
        v = self.vars["v"]
        q = self.vars["q"]
        I = self.vars["I"]
        d = self.data
        inst = self.instance

        sol["objective"] = float(self.model.ObjVal)
        sol["best_bound"] = float(self.model.ObjBound)
        if self.model.ObjVal != 0:
            sol["mip_gap"] = abs(self.model.ObjVal - self.model.ObjBound) / abs(self.model.ObjVal)
        else:
            sol["mip_gap"] = None
        sol["fleet_size"] = int(round(u.X))

        open_pls = []
        for i in d["pl_ids"]:
            selected_module = None
            selected_capacity = None
            selected_cost = None
            for m in d["module_ids"]:
                if z[i, m].X > 0.5:
                    selected_module = m
                    selected_capacity = d["module_cap"][m]
                    selected_cost = d["fixed_costs"][(i, m)]
                    break
            if selected_module is not None:
                open_pls.append(
                    {
                        "pl_id": i,
                        "module": selected_module,
                        "capacity": selected_capacity,
                        "fixed_cost": selected_cost,
                    }
                )
        sol["open_pls"] = open_pls

        assignments = {}
        for k in d["sr_ids"]:
            assigned = [i for i in d["pl_ids"] if y[i, k].X > 0.5]
            assignments[str(k)] = assigned[0] if assigned else None
        sol["assignments"] = assignments

        replenishment = {}
        inventory = {}
        for i in d["pl_ids"]:
            replenishment[str(i)] = {
                str(t): {
                    "visited": int(round(v[i, t].X)),
                    "quantity": float(q[i, t].X),
                }
                for t in d["periods"]
            }
            inventory[str(i)] = {str(t): float(I[i, t].X) for t in d["periods"]}
        sol["replenishment"] = replenishment
        sol["inventory"] = inventory

        x_vals = {(i, j, t): x[i, j, t].X for (i, j, t) in d["arcs"]}
        sol["routes"] = {str(t): routes for t, routes in self._route_reconstruction(x_vals).items()}

        fixed_pl_cost = sum(item["fixed_cost"] for item in open_pls)
        fleet_cost = inst.vehicle_fixed_cost * sol["fleet_size"]
        travel_cost = sum(inst.travel_cost * d["dist_nodes"][(i, j)] * x_vals[(i, j, t)] for (i, j, t) in d["arcs"])
        compensation_cost = sum(
            inst.compensation_cost * d["dist_assign"][(i, k)] * d["demand"][(k, t)] * y[i, k].X
            for i in d["pl_ids"] for k in d["sr_ids"] for t in d["periods"]
        )
        sol["objective_breakdown"] = {
            "fixed_pl_cost": fixed_pl_cost,
            "fleet_cost": fleet_cost,
            "travel_cost": travel_cost,
            "compensation_cost": compensation_cost,
        }
        sol["warm_start_input_summary"] = self._build_warm_start_summary(self.model.Runtime)
        if sol["warm_start_input_summary"].get("total_method_runtime_seconds") is not None:
            sol["total_method_runtime_seconds"] = sol["warm_start_input_summary"]["total_method_runtime_seconds"]
        else:
            sol["total_method_runtime_seconds"] = sol["runtime_seconds"]
        return sol


def solve_instance_file(
    instance_path: str | Path,
    output_path: Optional[str | Path] = None,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    no_vis: bool = False,
    threads: Optional[int] = None,
    warm_start_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    instance = PLLRPInstance.from_json(instance_path)
    solver = PLLRPBranchAndCut(
        instance,
        add_valid_inequalities=not no_vis,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        warm_start_path=warm_start_path,
    )
    solution = solver.solve()
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)
    return solution


def cli() -> None:
    parser = argparse.ArgumentParser(description="Branch-and-cut baseline for the PL-LRP paper model.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate a random PL-LRP instance similar to the paper settings.")
    g.add_argument("--output", required=True, help="Path to output JSON instance file.")
    g.add_argument("--n-pl", type=int, required=True)
    g.add_argument("--n-sr", type=int, required=True)
    g.add_argument("--n-modules", type=int, required=True)
    g.add_argument("--n-periods", type=int, required=True)
    g.add_argument("--pattern", choices=["uniform", "bimodal"], default="uniform")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--name", default=None)

    s = sub.add_parser("solve", help="Solve a PL-LRP instance with Gurobi branch-and-cut.")
    s.add_argument("--instance", required=True, help="Path to JSON instance file.")
    s.add_argument("--output", required=False, help="Optional path to JSON solution file.")
    s.add_argument("--time-limit", type=float, default=None)
    s.add_argument("--mip-gap", type=float, default=None)
    s.add_argument("--no-vis", action="store_true", help="Disable valid inequalities (16)-(21).")
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--warm-start", default=None, help="Optional path to a warm-start JSON file.")

    args = parser.parse_args()

    if args.command == "generate":
        inst = generate_random_instance(
            n_pl=args.n_pl,
            n_sr=args.n_sr,
            n_modules=args.n_modules,
            n_periods=args.n_periods,
            demand_pattern=args.pattern,
            seed=args.seed,
            name=args.name,
        )
        inst.to_json(args.output)
        print(f"Wrote instance to {args.output}")
        return

    if args.command == "solve":
        solution = solve_instance_file(
            instance_path=args.instance,
            output_path=args.output,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            no_vis=args.no_vis,
            threads=args.threads,
            warm_start_path=args.warm_start,
        )
        print(json.dumps(solution, indent=2))
        return


if __name__ == "__main__":
    cli()
