from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

gp: Any = None
GRB: Any = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=repr)]
    return value

try:
    gp = importlib.import_module("gurobipy")
    GRB = gp.GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

try:  # Prefer the latest report-capable LFRS module when available.
    import lfrs_warmstart as lfrs
except Exception:  # pragma: no cover
    try:
        import lfrs_warmstart as lfrs
    except Exception:  # pragma: no cover
        import lfrs_warmstart as lfrs


# ============================================================
# Data structures and instance utilities
# ============================================================


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
        candidate_pls = [
            CandidatePL(
                id=int(item["id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                fixed_costs={str(k): float(v) for k, v in item["fixed_costs"].items()},
            )
            for item in data["candidate_pls"]
        ]
        service_regions = [
            ServiceRegion(
                id=int(item["id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                demand={int(k): float(v) for k, v in item["demand"].items()},
            )
            for item in data["service_regions"]
        ]
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
                {"id": pl.id, "x": pl.x, "y": pl.y, "fixed_costs": pl.fixed_costs}
                for pl in self.candidate_pls
            ],
            "service_regions": [
                {"id": sr.id, "x": sr.x, "y": sr.y, "demand": sr.demand}
                for sr in self.service_regions
            ],
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


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


# ============================================================
# Model builder (same main MILP structure as the practical B&C
# baseline, reused by the VMND neighborhoods)
# ============================================================


class PLLRPModelBuilder:
    def __init__(
        self,
        instance: PLLRPInstance,
        add_valid_inequalities: bool = True,
        log_to_console: bool = True,
        numeric_focus: Optional[int] = 1,
        threads: Optional[int] = None,
    ) -> None:
        if gp is None or GRB is None:
            raise RuntimeError("gurobipy is not available. Install Gurobi and gurobipy first.")
        self.instance = instance
        self.add_valid_inequalities = add_valid_inequalities
        self.log_to_console = log_to_console
        self.numeric_focus = numeric_focus
        self.threads = threads
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
            remaining = sum(
                demand[(k, tau)] for k in sr_ids for tau in periods if period_pos[tau] >= period_pos[t]
            )
            mtilde[t] = min(inst.vehicle_capacity, max_module_capacity, remaining)

        d_min = min((demand[(k, t)] for k in sr_ids for t in periods), default=0.0)
        eps = 1e-6 if abs(d_min) < 1e-9 else 0.0
        lambda_first = {k: 1 if demand[(k, first_period)] > 0 else 0 for k in sr_ids}

        fixed_costs = {(pl.id, m): float(pl.fixed_costs[m]) for pl in inst.candidate_pls for m in module_ids}
        module_cap = {m: float(inst.modules[m]) for m in module_ids}

        self.data = {
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
        return self.data

    def build_model(
        self,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        name_suffix: str = "",
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        d = self._prepare_data()
        inst = self.instance

        model = gp.Model(f"{inst.name}{name_suffix}")
        model.Params.OutputFlag = 1 if self.log_to_console else 0
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)
        if mip_gap is not None:
            model.Params.MIPGap = float(mip_gap)
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

        fixed_pl_cost = gp.quicksum(
            d["fixed_costs"][(i, m)] * z[i, m] for i in d["pl_ids"] for m in d["module_ids"]
        )
        fleet_cost = inst.vehicle_fixed_cost * u
        travel_cost = gp.quicksum(
            inst.travel_cost * d["dist_nodes"][(i, j)] * x[i, j, t] for (i, j, t) in d["arcs"]
        )
        compensation_cost = gp.quicksum(
            inst.compensation_cost * d["dist_assign"][(i, k)] * d["demand"][(k, t)] * y[i, k]
            for i in d["pl_ids"] for k in d["sr_ids"] for t in d["periods"]
        )
        model.setObjective(fixed_pl_cost + fleet_cost + travel_cost + compensation_cost, GRB.MINIMIZE)

        model.addConstrs(
            (gp.quicksum(y[i, k] for i in d["pl_ids"]) == 1 for k in d["sr_ids"]),
            name="assign_each_sr",
        )

        model.addConstrs(
            (gp.quicksum(z[i, m] for m in d["module_ids"]) <= 1 for i in d["pl_ids"]),
            name="one_module_per_pl",
        )

        for i in d["pl_ids"]:
            cap_expr = gp.quicksum(d["module_cap"][m] * z[i, m] for m in d["module_ids"])
            for pos, t in enumerate(d["periods"]):
                prev_inv = 0.0 if pos == 0 else I[i, d["periods"][pos - 1]]
                demand_alloc = gp.quicksum(d["demand"][(k, t)] * y[i, k] for k in d["sr_ids"])
                model.addConstr(prev_inv + q[i, t] == demand_alloc + I[i, t], name=f"inv_bal_{i}_{t}")
                model.addConstr(demand_alloc + I[i, t] <= cap_expr, name=f"pl_cap_{i}_{t}")
                model.addConstr(
                    v[i, t] <= gp.quicksum(z[i, m] for m in d["module_ids"]),
                    name=f"visit_only_if_open_{i}_{t}",
                )
                model.addConstr(q[i, t] <= d["mtilde"][t] * v[i, t], name=f"qty_only_if_visit_{i}_{t}")

        model.addConstrs(
            (gp.quicksum(x[0, j, t] for j in d["pl_ids"]) <= u for t in d["periods"]),
            name="departures_le_fleet",
        )

        for i in d["pl_ids"]:
            for t in d["periods"]:
                out_expr = gp.quicksum(x[i, j, t] for j in d["nodes0prime"] if j != i)
                in_expr = gp.quicksum(x[j, i, t] for j in d["nodes0prime"] if j != i)
                model.addConstr(out_expr == in_expr, name=f"flow_balance_{i}_{t}")
                model.addConstr(out_expr == v[i, t], name=f"one_vehicle_per_pl_period_{i}_{t}")

        for i in d["pl_ids"]:
            for t in d["periods"]:
                model.addConstr(w[i, t] >= q[i, t], name=f"load_ge_delivery_{i}_{t}")
                model.addConstr(
                    w[i, t] <= inst.vehicle_capacity * v[i, t],
                    name=f"load_le_vehiclecap_{i}_{t}",
                )
        for (i, j, t) in d["pl_to_pl_arcs"]:
            model.addConstr(
                w[i, t] - w[j, t] >= q[i, t] - d["mtilde"][t] * (1 - x[i, j, t]),
                name=f"load_prop_{i}_{j}_{t}",
            )

        if self.add_valid_inequalities:
            for t in d["periods"]:
                lhs = gp.quicksum(
                    d["module_cap"][m] * z[i, m] for i in d["pl_ids"] for m in d["module_ids"]
                )
                rhs = gp.quicksum(d["demand"][(k, t)] for k in d["sr_ids"]) + gp.quicksum(I[i, t] for i in d["pl_ids"])
                model.addConstr(lhs >= rhs, name=f"vi_capacity_{t}")

            bound_by_module = {
                m: min(len(d["sr_ids"]), int(math.floor(d["module_cap"][m] / (d["d_min"] + d["eps"]))))
                for m in d["module_ids"]
            }
            for i in d["pl_ids"]:
                model.addConstr(
                    gp.quicksum(y[i, k] for k in d["sr_ids"]) <= gp.quicksum(
                        bound_by_module[m] * z[i, m] for m in d["module_ids"]
                    ),
                    name=f"vi_assign_card_{i}",
                )

            for i in d["pl_ids"]:
                open_expr = gp.quicksum(z[i, m] for m in d["module_ids"])
                for k in d["sr_ids"]:
                    model.addConstr(y[i, k] <= open_expr, name=f"vi_open_assign_{i}_{k}")

            ordered_periods = d["periods"]
            for idx, t in enumerate(ordered_periods):
                cumulative_departures = gp.quicksum(
                    x[0, j, tau] for tau in ordered_periods[: idx + 1] for j in d["pl_ids"]
                )
                cumulative_demand = sum(
                    d["demand"][(k, tau)] for k in d["sr_ids"] for tau in ordered_periods[: idx + 1]
                )
                model.addConstr(
                    cumulative_departures >= math.ceil(cumulative_demand / inst.vehicle_capacity),
                    name=f"vi_cum_depart_{t}",
                )

            for idx, t in enumerate(ordered_periods):
                cumulative_deliveries = gp.quicksum(q[i, tau] for i in d["pl_ids"] for tau in ordered_periods[: idx + 1])
                model.addConstr(
                    cumulative_deliveries <= (idx + 1) * inst.vehicle_capacity * u,
                    name=f"vi_cum_ship_{t}",
                )

            first = d["first_period"]
            for i in d["pl_ids"]:
                for k in d["sr_ids"]:
                    if d["lambda_first"][k] == 1:
                        model.addConstr(v[i, first] >= y[i, k], name=f"vi_first_period_{i}_{k}")

        model.update()
        vars_dict = {"z": z, "u": u, "y": y, "x": x, "v": v, "q": q, "w": w, "I": I}
        return model, vars_dict, d


# ============================================================
# Practical VMND solver
# ============================================================


class PLLRPVMND:
    """
    Practical VMND-style solver for the PL-LRP.

    This is not a literal reproduction of every implementation detail in the paper.
    It is a usable VMND variant built on top of the same practical MILP baseline:
      - heuristic feasible initial solution,
      - optional short global polishing run,
      - variable neighborhood descent with MIP neighborhoods,
      - sub-MIPs created by fixing all variables outside the selected neighborhood.

    Neighborhood families:
      1) assignment neighborhoods on subsets of service regions,
      2) location neighborhoods on subsets of parcel lockers,
      3) temporal neighborhoods on subsets of periods,
      4) route/replenishment neighborhoods on PL-period blocks.
    """

    def __init__(
        self,
        instance: PLLRPInstance,
        add_valid_inequalities: bool = True,
        log_to_console: bool = True,
        numeric_focus: Optional[int] = 1,
        threads: Optional[int] = None,
        seed: int = 0,
        max_iterations: int = 30,
        max_no_improve_rounds: int = 12,
        global_polish_time: float = 20.0,
        neighborhood_time: float = 15.0,
        neighborhood_gap: float = 0.02,
        improve_eps: float = 1e-4,
    ) -> None:
        self.instance = instance
        self.builder = PLLRPModelBuilder(
            instance=instance,
            add_valid_inequalities=add_valid_inequalities,
            log_to_console=log_to_console,
            numeric_focus=numeric_focus,
            threads=threads,
        )
        self.log_to_console = log_to_console
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_iterations = max_iterations
        self.max_no_improve_rounds = max_no_improve_rounds
        self.global_polish_time = global_polish_time
        self.neighborhood_time = neighborhood_time
        self.neighborhood_gap = neighborhood_gap
        self.improve_eps = improve_eps

    # --------------------------------------------------------
    # incumbent handling
    # --------------------------------------------------------

    def _empty_incumbent(self, d: Dict[str, Any]) -> Dict[str, Dict[Any, float]]:
        return {
            "z": {(i, m): 0.0 for i in d["pl_ids"] for m in d["module_ids"]},
            "u": {None: 0.0},
            "y": {(i, k): 0.0 for i in d["pl_ids"] for k in d["sr_ids"]},
            "x": {(i, j, t): 0.0 for (i, j, t) in d["arcs"]},
            "v": {(i, t): 0.0 for i in d["pl_ids"] for t in d["periods"]},
            "q": {(i, t): 0.0 for i in d["pl_ids"] for t in d["periods"]},
            "w": {(i, t): 0.0 for i in d["pl_ids"] for t in d["periods"]},
            "I": {(i, t): 0.0 for i in d["pl_ids"] for t in d["periods"]},
        }

    def _compute_objective(self, incumbent: Dict[str, Dict[Any, float]], d: Dict[str, Any]) -> float:
        inst = self.instance
        fixed_pl_cost = sum(
            d["fixed_costs"][(i, m)] * incumbent["z"][(i, m)] for i in d["pl_ids"] for m in d["module_ids"]
        )
        fleet_cost = inst.vehicle_fixed_cost * incumbent["u"][None]
        travel_cost = sum(
            inst.travel_cost * d["dist_nodes"][(i, j)] * incumbent["x"][(i, j, t)]
            for (i, j, t) in d["arcs"]
        )
        compensation_cost = sum(
            inst.compensation_cost * d["dist_assign"][(i, k)] * d["demand"][(k, t)] * incumbent["y"][(i, k)]
            for i in d["pl_ids"] for k in d["sr_ids"] for t in d["periods"]
        )
        return fixed_pl_cost + fleet_cost + travel_cost + compensation_cost

    def _extract_incumbent(self, model: Any, vars_dict: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, Dict[Any, float]]:
        inc = self._empty_incumbent(d)
        if model.SolCount == 0:
            return inc
        for i in d["pl_ids"]:
            for m in d["module_ids"]:
                inc["z"][(i, m)] = round(vars_dict["z"][i, m].X)
        inc["u"][None] = round(vars_dict["u"].X)
        for i in d["pl_ids"]:
            for k in d["sr_ids"]:
                inc["y"][(i, k)] = round(vars_dict["y"][i, k].X)
        for (i, j, t) in d["arcs"]:
            inc["x"][(i, j, t)] = round(vars_dict["x"][i, j, t].X)
        for i in d["pl_ids"]:
            for t in d["periods"]:
                inc["v"][(i, t)] = round(vars_dict["v"][i, t].X)
                inc["q"][(i, t)] = float(vars_dict["q"][i, t].X)
                inc["w"][(i, t)] = float(vars_dict["w"][i, t].X)
                inc["I"][(i, t)] = float(vars_dict["I"][i, t].X)
        return inc

    def _apply_start(self, vars_dict: Dict[str, Any], incumbent: Dict[str, Dict[Any, float]], d: Dict[str, Any]) -> None:
        for i in d["pl_ids"]:
            for m in d["module_ids"]:
                vars_dict["z"][i, m].Start = incumbent["z"][(i, m)]
        vars_dict["u"].Start = incumbent["u"][None]
        for i in d["pl_ids"]:
            for k in d["sr_ids"]:
                vars_dict["y"][i, k].Start = incumbent["y"][(i, k)]
        for (i, j, t) in d["arcs"]:
            vars_dict["x"][i, j, t].Start = incumbent["x"][(i, j, t)]
        for i in d["pl_ids"]:
            for t in d["periods"]:
                vars_dict["v"][i, t].Start = incumbent["v"][(i, t)]
                vars_dict["q"][i, t].Start = incumbent["q"][(i, t)]
                vars_dict["w"][i, t].Start = incumbent["w"][(i, t)]
                vars_dict["I"][i, t].Start = incumbent["I"][(i, t)]

    def _set_fixed(self, var: Any, value: float) -> None:
        if var.VType in (GRB.BINARY, GRB.INTEGER):
            val = int(round(value))
            var.LB = val
            var.UB = val
        else:
            var.LB = float(value)
            var.UB = float(value)

    def _set_free(self, var: Any, lower: Optional[float] = None, upper: Optional[float] = None) -> None:
        if lower is None:
            lower = 0.0
        if upper is None:
            upper = GRB.INFINITY
        var.LB = lower
        var.UB = upper

    # --------------------------------------------------------
    # constructive initial solution
    # --------------------------------------------------------

    def _heuristic_initial_solution(self) -> Tuple[Dict[str, Dict[Any, float]], Dict[str, Any]]:
        d = self.builder._prepare_data()

        inst = self.instance
        incumbent = self._empty_incumbent(d)
        periods = d["periods"]
        cap_bound = min(max(d["module_cap"].values()), inst.vehicle_capacity)

        assigned_to_pl: Dict[int, List[int]] = {i: [] for i in d["pl_ids"]}
        profile_by_pl: Dict[int, Dict[int, float]] = {i: {t: 0.0 for t in periods} for i in d["pl_ids"]}
        open_pls: Set[int] = set()

        def min_opening_cost(i: int) -> float:
            return min(d["fixed_costs"][(i, m)] for m in d["module_ids"])

        sr_order = sorted(
            d["sr_ids"],
            key=lambda k: sum(d["demand"][(k, t)] for t in periods),
            reverse=True,
        )

        for k in sr_order:
            best_choice: Optional[Tuple[float, int]] = None
            sr_profile = {t: d["demand"][(k, t)] for t in periods}
            for i in d["pl_ids"]:
                feasible = True
                for t in periods:
                    if profile_by_pl[i][t] + sr_profile[t] > cap_bound + 1e-9:
                        feasible = False
                        break
                if not feasible:
                    continue
                comp = sum(
                    inst.compensation_cost * d["dist_assign"][(i, k)] * d["demand"][(k, t)]
                    for t in periods
                )
                extra_opening = 0.0 if i in open_pls else min_opening_cost(i)
                score = comp + extra_opening
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, i)
            if best_choice is None:
                raise RuntimeError(
                    "Heuristic could not build a feasible assignment. Check vehicle/module capacities versus demand."
                )
            i = best_choice[1]
            open_pls.add(i)
            assigned_to_pl[i].append(k)
            for t in periods:
                profile_by_pl[i][t] += sr_profile[t]
                incumbent["y"][(i, k)] = 1.0

        for i in open_pls:
            peak = max(profile_by_pl[i][t] for t in periods)
            chosen_module = None
            for m in sorted(d["module_ids"], key=lambda mm: d["module_cap"][mm]):
                if d["module_cap"][m] + 1e-9 >= peak:
                    chosen_module = m
                    break
            if chosen_module is None:
                raise RuntimeError(f"No feasible module can support locker {i} peak profile {peak}.")
            incumbent["z"][(i, chosen_module)] = 1.0

        max_routes_in_period = 0
        for i in open_pls:
            for t in periods:
                qty = profile_by_pl[i][t]
                incumbent["q"][(i, t)] = qty
                incumbent["I"][(i, t)] = 0.0
                incumbent["w"][(i, t)] = qty
                if qty > 1e-9:
                    incumbent["v"][(i, t)] = 1.0
                    incumbent["x"][(0, i, t)] = 1.0
                    incumbent["x"][(i, 0, t)] = 1.0
        for t in periods:
            max_routes_in_period = max(
                max_routes_in_period,
                int(sum(incumbent["v"][(i, t)] for i in open_pls)),
            )
        incumbent["u"][None] = float(max_routes_in_period)

        sol = self._incumbent_to_solution_dict(incumbent, d)
        sol["initial_solution_type"] = "greedy_feasible"
        return incumbent, sol

    # --------------------------------------------------------
    # Neighborhood generation helpers
    # --------------------------------------------------------

    def _current_assignment(self, incumbent: Dict[str, Dict[Any, float]], d: Dict[str, Any]) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for k in d["sr_ids"]:
            for i in d["pl_ids"]:
                if incumbent["y"][(i, k)] > 0.5:
                    mapping[k] = i
                    break
        return mapping

    def _current_open_pls(self, incumbent: Dict[str, Dict[Any, float]], d: Dict[str, Any]) -> Set[int]:
        return {
            i for i in d["pl_ids"]
            if sum(incumbent["z"][(i, m)] for m in d["module_ids"]) > 0.5
        }

    def _busiest_pls(self, incumbent: Dict[str, Dict[Any, float]], d: Dict[str, Any], count: int) -> List[int]:
        scored = []
        for i in d["pl_ids"]:
            total = sum(incumbent["q"][(i, t)] for t in d["periods"])
            if total > 1e-9:
                scored.append((total, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:count]]

    def _high_cost_srs(self, incumbent: Dict[str, Dict[Any, float]], d: Dict[str, Any], count: int) -> List[int]:
        current = self._current_assignment(incumbent, d)
        scored = []
        for k, i in current.items():
            comp = sum(
                self.instance.compensation_cost * d["dist_assign"][(i, k)] * d["demand"][(k, t)]
                for t in d["periods"]
            )
            scored.append((comp, k))
        scored.sort(reverse=True)
        return [k for _, k in scored[:count]]

    def _peak_periods(self, incumbent: Dict[str, Dict[Any, float]], d: Dict[str, Any], count: int) -> List[int]:
        scored = []
        for t in d["periods"]:
            total = sum(incumbent["q"][(i, t)] for i in d["pl_ids"])
            scored.append((total, t))
        scored.sort(reverse=True)
        return [t for _, t in scored[:count]]

    def _related_pls_for_srs(
        self,
        incumbent: Dict[str, Dict[Any, float]],
        d: Dict[str, Any],
        sr_subset: Sequence[int],
        k_nearest: int = 4,
    ) -> Set[int]:
        current = self._current_assignment(incumbent, d)
        related: Set[int] = set()
        for k in sr_subset:
            if k in current:
                related.add(current[k])
            nearest = sorted(d["pl_ids"], key=lambda i: d["dist_assign"][(i, k)])[:k_nearest]
            related.update(nearest)
        return related

    # --------------------------------------------------------
    # Fixings for MIP neighborhoods
    # --------------------------------------------------------

    def _apply_default_fixing(
        self,
        vars_dict: Dict[str, Any],
        incumbent: Dict[str, Dict[Any, float]],
        d: Dict[str, Any],
    ) -> None:
        for i in d["pl_ids"]:
            for m in d["module_ids"]:
                self._set_fixed(vars_dict["z"][i, m], incumbent["z"][(i, m)])
        self._set_fixed(vars_dict["u"], incumbent["u"][None])
        for i in d["pl_ids"]:
            for k in d["sr_ids"]:
                self._set_fixed(vars_dict["y"][i, k], incumbent["y"][(i, k)])
        for (i, j, t) in d["arcs"]:
            self._set_fixed(vars_dict["x"][i, j, t], incumbent["x"][(i, j, t)])
        for i in d["pl_ids"]:
            for t in d["periods"]:
                self._set_fixed(vars_dict["v"][i, t], incumbent["v"][(i, t)])
                self._set_fixed(vars_dict["q"][i, t], incumbent["q"][(i, t)])
                self._set_fixed(vars_dict["w"][i, t], incumbent["w"][(i, t)])
                self._set_fixed(vars_dict["I"][i, t], incumbent["I"][(i, t)])

    def _free_assignment_neighborhood(
        self,
        vars_dict: Dict[str, Any],
        incumbent: Dict[str, Dict[Any, float]],
        d: Dict[str, Any],
        sr_subset: Sequence[int],
    ) -> None:
        related_pls = self._related_pls_for_srs(incumbent, d, sr_subset)
        current = self._current_assignment(incumbent, d)
        for k in sr_subset:
            for i in d["pl_ids"]:
                self._set_free(vars_dict["y"][i, k], 0.0, 1.0)
        for i in related_pls:
            for m in d["module_ids"]:
                self._set_free(vars_dict["z"][i, m], 0.0, 1.0)
            for t in d["periods"]:
                self._set_free(vars_dict["v"][i, t], 0.0, 1.0)
                self._set_free(vars_dict["q"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["w"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["I"][i, t], 0.0, GRB.INFINITY)
        for (i, j, t) in d["arcs"]:
            if i in related_pls or j in related_pls:
                self._set_free(vars_dict["x"][i, j, t], 0.0, 1.0)
        impacted_routes = set(related_pls)
        for k, pl in current.items():
            if pl in related_pls:
                impacted_routes.add(pl)
        max_v = max(1, int(round(incumbent["u"][None])) + len(sr_subset))
        self._set_free(vars_dict["u"], 0.0, max_v)

    def _free_pl_neighborhood(
        self,
        vars_dict: Dict[str, Any],
        incumbent: Dict[str, Dict[Any, float]],
        d: Dict[str, Any],
        pl_subset: Sequence[int],
    ) -> None:
        pl_set = set(pl_subset)
        current = self._current_assignment(incumbent, d)
        impacted_srs = [k for k, i in current.items() if i in pl_set]
        for i in pl_set:
            for m in d["module_ids"]:
                self._set_free(vars_dict["z"][i, m], 0.0, 1.0)
            for t in d["periods"]:
                self._set_free(vars_dict["v"][i, t], 0.0, 1.0)
                self._set_free(vars_dict["q"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["w"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["I"][i, t], 0.0, GRB.INFINITY)
        for k in impacted_srs:
            for i in d["pl_ids"]:
                self._set_free(vars_dict["y"][i, k], 0.0, 1.0)
        for (i, j, t) in d["arcs"]:
            if i in pl_set or j in pl_set:
                self._set_free(vars_dict["x"][i, j, t], 0.0, 1.0)
        max_v = max(1, int(round(incumbent["u"][None])) + len(pl_subset))
        self._set_free(vars_dict["u"], 0.0, max_v)

    def _free_time_neighborhood(
        self,
        vars_dict: Dict[str, Any],
        incumbent: Dict[str, Dict[Any, float]],
        d: Dict[str, Any],
        periods_subset: Sequence[int],
    ) -> None:
        tset = set(periods_subset)
        for i in d["pl_ids"]:
            for t in tset:
                self._set_free(vars_dict["v"][i, t], 0.0, 1.0)
                self._set_free(vars_dict["q"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["w"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["I"][i, t], 0.0, GRB.INFINITY)
        for (i, j, t) in d["arcs"]:
            if t in tset:
                self._set_free(vars_dict["x"][i, j, t], 0.0, 1.0)
        max_v = max(1, int(round(incumbent["u"][None])) + len(periods_subset))
        self._set_free(vars_dict["u"], 0.0, max_v)

    def _free_block_neighborhood(
        self,
        vars_dict: Dict[str, Any],
        incumbent: Dict[str, Dict[Any, float]],
        d: Dict[str, Any],
        pl_subset: Sequence[int],
        periods_subset: Sequence[int],
    ) -> None:
        pl_set = set(pl_subset)
        tset = set(periods_subset)
        for i in pl_set:
            for t in tset:
                self._set_free(vars_dict["v"][i, t], 0.0, 1.0)
                self._set_free(vars_dict["q"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["w"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["I"][i, t], 0.0, GRB.INFINITY)
        for (i, j, t) in d["arcs"]:
            if t in tset and (i in pl_set or j in pl_set):
                self._set_free(vars_dict["x"][i, j, t], 0.0, 1.0)
        max_v = max(1, int(round(incumbent["u"][None])) + len(pl_subset))
        self._set_free(vars_dict["u"], 0.0, max_v)

    # --------------------------------------------------------
    # Neighborhood solve wrapper
    # --------------------------------------------------------

    def _solve_submip(
        self,
        incumbent: Dict[str, Dict[Any, float]],
        free_callback,
        label: str,
        data_args: Tuple[Any, ...],
    ) -> Optional[Tuple[Dict[str, Dict[Any, float]], Dict[str, Any]]]:
        model, vars_dict, d = self.builder.build_model(
            time_limit=self.neighborhood_time,
            mip_gap=self.neighborhood_gap,
            name_suffix=f"_{label}",
        )
        self._apply_start(vars_dict, incumbent, d)
        self._apply_default_fixing(vars_dict, incumbent, d)
        free_callback(vars_dict, incumbent, d, *data_args)
        incumbent_obj = self._compute_objective(incumbent, d)
        model.addConstr(model.getObjective() <= incumbent_obj - self.improve_eps, name=f"improve_{label}")
        model.update()
        model.optimize()

        if model.SolCount == 0:
            return None
        new_inc = self._extract_incumbent(model, vars_dict, d)
        new_obj = self._compute_objective(new_inc, d)
        if new_obj + 1e-9 < incumbent_obj:
            return new_inc, self._incumbent_to_solution_dict(new_inc, d, runtime=model.Runtime)
        return None

    # --------------------------------------------------------
    # VMND main loop
    # --------------------------------------------------------

    def solve(self) -> Dict[str, Any]:
        incumbent, best_sol = self._heuristic_initial_solution()
        d = self.builder._prepare_data()
        best_obj = self._compute_objective(incumbent, d)

        if self.log_to_console:
            print(f"[VMND] Initial heuristic objective = {best_obj:.4f}")

        # Optional short global polish from the heuristic start.
        if self.global_polish_time > 1e-6:
            model, vars_dict, d = self.builder.build_model(
                time_limit=self.global_polish_time,
                mip_gap=0.02,
                name_suffix="_global_polish",
            )
            self._apply_start(vars_dict, incumbent, d)
            model.optimize()
            if model.SolCount > 0:
                polished = self._extract_incumbent(model, vars_dict, d)
                polished_obj = self._compute_objective(polished, d)
                if polished_obj + 1e-9 < best_obj:
                    incumbent = polished
                    best_obj = polished_obj
                    best_sol = self._incumbent_to_solution_dict(polished, d, runtime=model.Runtime)
                    best_sol["initial_solution_type"] = "heuristic_plus_short_global_polish"
                    if self.log_to_console:
                        print(f"[VMND] Global polish improved incumbent to {best_obj:.4f}")

        no_improve_rounds = 0
        iteration = 0
        improvement_log: List[Dict[str, Any]] = []

        while iteration < self.max_iterations and no_improve_rounds < self.max_no_improve_rounds:
            iteration += 1
            improved = False
            current_assign = self._current_assignment(incumbent, d)
            current_open = self._current_open_pls(incumbent, d)

            assign_sets: List[List[int]] = []
            if d["sr_ids"]:
                heavy = self._high_cost_srs(incumbent, d, min(4, len(d["sr_ids"])))
                if heavy:
                    assign_sets.append(heavy[: min(2, len(heavy))])
                    assign_sets.append(heavy)
                shuffled_srs = list(d["sr_ids"])
                self.rng.shuffle(shuffled_srs)
                assign_sets.append(shuffled_srs[: min(3, len(shuffled_srs))])

            pl_sets: List[List[int]] = []
            busy = self._busiest_pls(incumbent, d, min(5, len(d["pl_ids"])))
            if busy:
                pl_sets.append(busy[: min(2, len(busy))])
                pl_sets.append(busy[: min(4, len(busy))])
            open_list = list(current_open)
            self.rng.shuffle(open_list)
            if open_list:
                pl_sets.append(open_list[: min(3, len(open_list))])

            time_sets: List[List[int]] = []
            peaks = self._peak_periods(incumbent, d, min(3, len(d["periods"])))
            if peaks:
                time_sets.append(peaks[:1])
                time_sets.append(peaks)
            if len(d["periods"]) >= 2:
                time_sets.append(d["periods"][:2])

            block_sets: List[Tuple[List[int], List[int]]] = []
            if busy and peaks:
                block_sets.append((busy[: min(2, len(busy))], peaks[:1]))
                block_sets.append((busy[: min(3, len(busy))], peaks[: min(2, len(peaks))]))

            neighborhoods: List[Tuple[str, Any]] = []
            neighborhoods.extend(("assign", s) for s in assign_sets if s)
            neighborhoods.extend(("pl", s) for s in pl_sets if s)
            neighborhoods.extend(("time", s) for s in time_sets if s)
            neighborhoods.extend(("block", s) for s in block_sets if s[0] and s[1])

            for ntype, payload in neighborhoods:
                if ntype == "assign":
                    result = self._solve_submip(incumbent, self._free_assignment_neighborhood, "assign", (payload,))
                elif ntype == "pl":
                    result = self._solve_submip(incumbent, self._free_pl_neighborhood, "pl", (payload,))
                elif ntype == "time":
                    result = self._solve_submip(incumbent, self._free_time_neighborhood, "time", (payload,))
                else:
                    pls, periods = payload
                    result = self._solve_submip(incumbent, self._free_block_neighborhood, "block", (pls, periods))

                if result is None:
                    continue

                new_inc, new_sol = result
                new_obj = self._compute_objective(new_inc, d)
                if new_obj + 1e-9 < best_obj:
                    incumbent = new_inc
                    best_obj = new_obj
                    best_sol = new_sol
                    improvement_log.append(
                        {
                            "iteration": iteration,
                            "neighborhood": ntype,
                            "payload": payload,
                            "objective": best_obj,
                        }
                    )
                    if self.log_to_console:
                        print(f"[VMND] Iter {iteration}: {ntype} neighborhood improved objective to {best_obj:.4f}")
                    improved = True
                    break

            if improved:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if self.log_to_console:
                    print(f"[VMND] Iter {iteration}: no improvement (round {no_improve_rounds}).")

        best_sol["algorithm"] = "VMND"
        best_sol["vmnd_iterations"] = iteration
        best_sol["vmnd_no_improve_rounds"] = no_improve_rounds
        best_sol["vmnd_improvement_log"] = improvement_log
        best_sol["seed"] = self.seed
        return best_sol

    # --------------------------------------------------------
    # solution formatting
    # --------------------------------------------------------

    def _route_reconstruction(self, x_vals: Dict[Tuple[int, int, int], float], d: Dict[str, Any]) -> Dict[int, List[List[int]]]:
        periods = d["periods"]
        pl_ids = set(d["pl_ids"])
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
                    if nxt == 0 or nxt in visited:
                        break
                    visited.add(nxt)
                    cur = nxt
                routes[t].append(route)
        return routes

    def _incumbent_to_solution_dict(
        self,
        incumbent: Dict[str, Dict[Any, float]],
        d: Dict[str, Any],
        runtime: float = 0.0,
    ) -> Dict[str, Any]:
        inst = self.instance
        assignments = {}
        for k in d["sr_ids"]:
            assigned = None
            for i in d["pl_ids"]:
                if incumbent["y"][(i, k)] > 0.5:
                    assigned = i
                    break
            assignments[str(k)] = assigned

        open_pls = []
        for i in d["pl_ids"]:
            selected_module = None
            for m in d["module_ids"]:
                if incumbent["z"][(i, m)] > 0.5:
                    selected_module = m
                    break
            if selected_module is not None:
                open_pls.append(
                    {
                        "pl_id": i,
                        "module": selected_module,
                        "capacity": d["module_cap"][selected_module],
                        "fixed_cost": d["fixed_costs"][(i, selected_module)],
                    }
                )

        replenishment = {}
        inventory = {}
        for i in d["pl_ids"]:
            replenishment[str(i)] = {
                str(t): {
                    "visited": int(round(incumbent["v"][(i, t)])),
                    "quantity": float(incumbent["q"][(i, t)]),
                }
                for t in d["periods"]
            }
            inventory[str(i)] = {str(t): float(incumbent["I"][(i, t)]) for t in d["periods"]}

        routes = self._route_reconstruction(incumbent["x"], d)
        fixed_pl_cost = sum(item["fixed_cost"] for item in open_pls)
        fleet_cost = inst.vehicle_fixed_cost * incumbent["u"][None]
        travel_cost = sum(
            inst.travel_cost * d["dist_nodes"][(i, j)] * incumbent["x"][(i, j, t)]
            for (i, j, t) in d["arcs"]
        )
        compensation_cost = sum(
            inst.compensation_cost * d["dist_assign"][(i, k)] * d["demand"][(k, t)] * incumbent["y"][(i, k)]
            for i in d["pl_ids"] for k in d["sr_ids"] for t in d["periods"]
        )

        return {
            "instance": self.instance.name,
            "status": "FEASIBLE",
            "objective": fixed_pl_cost + fleet_cost + travel_cost + compensation_cost,
            "best_bound": None,
            "mip_gap": None,
            "runtime_seconds": runtime,
            "open_pls": open_pls,
            "fleet_size": int(round(incumbent["u"][None])),
            "assignments": assignments,
            "replenishment": replenishment,
            "inventory": inventory,
            "routes": {str(t): routes[t] for t in d["periods"]},
            "objective_breakdown": {
                "fixed_pl_cost": fixed_pl_cost,
                "fleet_cost": fleet_cost,
                "travel_cost": travel_cost,
                "compensation_cost": compensation_cost,
            },
        }

# ============================================================
# Paper parameters
# ============================================================


@dataclass
class PaperPLLRPParams:
    fixed_vehicle_cost: float = 50_000.0
    alpha: float = 1.37
    beta: float = 0.11
    vehicle_capacity: float = 400.0
    side_length: float = 25.0
    depot_x: float = 12.5
    depot_y: float = 12.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperVMNDParams:
    mu_V: float = 10800.0
    mu_B_min: float = 120.0
    mu_B_max: float = 360.0
    mu_L: float = 300.0
    gamma_min: float = 0.0005
    gamma_max: float = 0.01
    operator_order: Tuple[int, ...] = (4, 5, 3, 2, 1)
    mu_I: float = 600.0
    xi2: float = 0.35
    xi3: float = 0.35
    xi4: float = 0.50
    xi5: float = 0.35
    seed: int = 0
    neighborhood_gap: float = 0.02
    improve_eps: float = 1e-4
    threads: Optional[int] = None
    numeric_focus: Optional[int] = 1
    use_valid_inequalities: bool = True
    max_stagnant_bcp_rounds: int = 999999
    stop_if_bcp_repeats_incumbent: bool = False
    force_lsp_after_bcp_no_improve: bool = True

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PaperVMNDParams":
        obj = PaperVMNDParams()
        for k, v in data.items():
            if not hasattr(obj, k):
                continue
            if k == "operator_order":
                setattr(obj, k, tuple(int(x) for x in v))
            else:
                setattr(obj, k, v)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["operator_order"] = list(self.operator_order)
        return out

# ============================================================
# Shared instance loaders / LFRS bridge
# ============================================================


def convert_lfrs_instance(inst: lfrs.PLLRPInstance, name: str = "pl_lrp_instance") -> PLLRPInstance:
    depot_xy = inst.coords[inst.depot_id]
    depot = Depot(float(depot_xy[0]), float(depot_xy[1]))

    candidate_pls = []
    for i in inst.pl_ids:
        fixed_costs = {str(m): float(inst.opening_cost[(i, m)]) for m in inst.module_ids}
        candidate_pls.append(CandidatePL(id=int(i), x=float(inst.coords[i][0]), y=float(inst.coords[i][1]), fixed_costs=fixed_costs))

    service_regions = []
    for k in inst.sr_ids:
        if k in inst.coords:
            xk, yk = inst.coords[k]
        else:
            xk, yk = depot_xy
        demand = {int(t): float(inst.demand[(k, t)]) for t in inst.periods}
        service_regions.append(ServiceRegion(id=int(k), x=float(xk), y=float(yk), demand=demand))

    return PLLRPInstance(
        periods=[int(t) for t in inst.periods],
        modules={str(m): float(inst.module_capacity[m]) for m in inst.module_ids},
        vehicle_capacity=float(inst.vehicle_capacity),
        vehicle_fixed_cost=float(inst.vehicle_fixed_cost),
        travel_cost=float(inst.alpha),
        compensation_cost=float(inst.beta),
        depot=depot,
        candidate_pls=candidate_pls,
        service_regions=service_regions,
        name=name,
    )


def load_instance_any(path: str | Path) -> PLLRPInstance:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if {'candidate_pls', 'service_regions', 'modules', 'depot'}.issubset(set(data.keys())):
        return PLLRPInstance.from_dict(data)

    lfrs_inst = lfrs.build_instance_from_plain_dict(data)
    return convert_lfrs_instance(lfrs_inst, str(data.get('name', 'pl_lrp_instance')))


# ============================================================
# Warm-start file I/O
# ============================================================


def _parse_scalar_key(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("m") and len(v) > 1 and v[1:].isdigit():
            return int(v[1:])
        return int(v)
    raise ValueError(f"Unsupported scalar key: {value!r}")


def _iter_records(obj: Any, expected_dim: int) -> Iterable[Tuple[Tuple[int, ...], float]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        out = []
        for rec in obj:
            if isinstance(rec, dict):
                if "key" in rec:
                    key_raw = rec["key"]
                    if not isinstance(key_raw, (list, tuple)) or len(key_raw) != expected_dim:
                        raise ValueError(f"Expected key length {expected_dim}, got {rec}")
                    key = tuple(_parse_scalar_key(x) for x in key_raw)
                    value = float(rec.get("value", 0))
                    out.append((key, value))
                    continue
                if expected_dim == 2:
                    if {"pl_id", "module"} <= set(rec.keys()):
                        out.append(((int(rec["pl_id"]), _parse_scalar_key(rec["module"])), float(rec.get("value", 1))))
                    elif {"pl_id", "sr_id"} <= set(rec.keys()):
                        out.append(((int(rec["pl_id"]), int(rec["sr_id"])), float(rec.get("value", 1))))
                    elif {"pl_id", "period"} <= set(rec.keys()):
                        out.append(((int(rec["pl_id"]), int(rec["period"])), float(rec.get("value", 0))))
                    else:
                        raise ValueError(f"Unsupported 2D record: {rec}")
                elif expected_dim == 3:
                    out.append(((int(rec["from"]), int(rec["to"]), int(rec["period"])), float(rec.get("value", 1))))
                else:
                    raise ValueError(f"Unsupported expected_dim={expected_dim}")
            else:
                if len(rec) != expected_dim + 1:
                    raise ValueError(f"Expected tuple/list len {expected_dim + 1}, got {rec}")
                key = tuple(_parse_scalar_key(rec[i]) for i in range(expected_dim))
                out.append((key, float(rec[expected_dim])))
        return out

    if isinstance(obj, Mapping):
        out = []
        for k, v in obj.items():
            if isinstance(k, str):
                norm = k.strip()
                if norm.startswith("(") and norm.endswith(")"):
                    try:
                        parsed = ast.literal_eval(norm)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, (tuple, list)) and len(parsed) == expected_dim:
                        key = tuple(_parse_scalar_key(x) for x in parsed)
                    else:
                        parts = [p.strip() for p in norm.strip("()").replace("|", ",").split(",") if p.strip()]
                        if len(parts) != expected_dim:
                            raise ValueError(f"Unsupported key format: {k}")
                        key = tuple(_parse_scalar_key(x) for x in parts)
                else:
                    parts = [p.strip() for p in norm.replace("|", ",").split(",") if p.strip()]
                    if len(parts) != expected_dim:
                        raise ValueError(f"Unsupported key format: {k}")
                    key = tuple(_parse_scalar_key(x) for x in parts)
            elif isinstance(k, (tuple, list)) and len(k) == expected_dim:
                key = tuple(_parse_scalar_key(x) for x in k)
            else:
                raise ValueError(f"Unsupported key type: {k}")
            out.append((key, float(v)))
        return out

    raise ValueError("Unsupported record container")


def load_warm_start_payload(path: str | Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError("Warm-start JSON must be a mapping.")
    if "warm_start" in data and isinstance(data["warm_start"], Mapping):
        return data["warm_start"]
    return data


def load_warm_start_file(path: str | Path, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    data = payload if payload is not None else load_warm_start_payload(path)
    starts_root = data.get("starts", data) if isinstance(data, Mapping) else data
    if not isinstance(starts_root, Mapping):
        raise ValueError("Warm-start JSON must be a mapping or contain a top-level 'starts' mapping.")

    parsed = {
        "z": {k: val for k, val in _iter_records(starts_root.get("z"), 2)},
        "y": {k: val for k, val in _iter_records(starts_root.get("y"), 2)},
        "v": {k: val for k, val in _iter_records(starts_root.get("v"), 2)},
        "q": {k: val for k, val in _iter_records(starts_root.get("q"), 2)},
        "I": {k: val for k, val in _iter_records(starts_root.get("I"), 2)},
        "x": {k: val for k, val in _iter_records(starts_root.get("x"), 3)},
        "w": {k: val for k, val in _iter_records(starts_root.get("w"), 2)},
        "u": 0,
        "period_routes": {},
        "meta": data.get("meta", {}) if isinstance(data, Mapping) else {},
        "diagnostics": data.get("diagnostics", {}) if isinstance(data, Mapping) else {},
        "result_summary": data.get("result_summary", {}) if isinstance(data, Mapping) else {},
    }

    uobj = starts_root.get("u", 0)
    if isinstance(uobj, Mapping):
        parsed["u"] = int(round(float(uobj.get("u", uobj.get("value", 0)))))
    else:
        parsed["u"] = int(round(float(uobj or 0)))

    pr = data.get("period_routes")
    if pr is None and isinstance(data.get("diagnostics"), Mapping):
        pr = data["diagnostics"].get("period_routes", {})
    if isinstance(pr, Mapping):
        parsed["period_routes"] = {int(t): [list(map(int, r)) for r in routes] for t, routes in pr.items()}
    return parsed


def _instance_meta_for_vmnd(instance: PLLRPInstance, source: str = "VMND", schema_version: str = "1.0") -> Dict[str, Any]:
    return {
        "instance_name": instance.name,
        "source": source,
        "schema_version": schema_version,
        "periods": [int(t) for t in instance.periods],
        "module_ids": sorted(int(k) for k in instance.modules.keys()),
        "pl_ids": sorted(int(pl.id) for pl in instance.candidate_pls),
        "sr_ids": sorted(int(sr.id) for sr in instance.service_regions),
        "vehicle_capacity": float(instance.vehicle_capacity),
        "vehicle_fixed_cost": float(instance.vehicle_fixed_cost),
    }


def _serialize_start_section(section: Mapping[Any, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for key, value in section.items():
        key_list = list(key) if isinstance(key, tuple) else [key]
        records.append({"key": key_list, "value": value})
    records.sort(key=lambda rec: tuple(rec["key"]))
    return records


def export_warm_start_file(
    path: str | Path,
    instance_name: str,
    warm: Dict[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    result_summary = dict(extra.get("result_summary", {})) if isinstance(extra, Mapping) and isinstance(extra.get("result_summary"), Mapping) else {}
    source = str(extra.get("source", "VMND")) if isinstance(extra, Mapping) else "VMND"
    schema_version = str(extra.get("schema_version", "1.0")) if isinstance(extra, Mapping) else "1.0"
    diagnostics_extra = dict(extra.get("diagnostics", {})) if isinstance(extra, Mapping) and isinstance(extra.get("diagnostics"), Mapping) else {}
    payload = {
        "meta": {
            "instance_name": instance_name,
            "source": source,
            "schema_version": schema_version,
        },
        "starts": {
            "z": _serialize_start_section(warm.get("z", {})),
            "y": _serialize_start_section(warm.get("y", {})),
            "v": _serialize_start_section(warm.get("v", {})),
            "q": _serialize_start_section(warm.get("q", {})),
            "I": _serialize_start_section(warm.get("I", {})),
            "x": _serialize_start_section(warm.get("x", {})),
            "w": _serialize_start_section(warm.get("w", {})),
            "u": {"u": int(round(warm.get("u", 0)))},
        },
        "diagnostics": {
            "period_routes": {str(t): routes for t, routes in warm.get("period_routes", {}).items()},
            **diagnostics_extra,
        },
    }
    if result_summary:
        payload["result_summary"] = result_summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Paper-style VMND wrapper
# ============================================================


class PLLRPVMNDPaper(PLLRPVMND):
    def __init__(
        self,
        instance: PLLRPInstance,
        params: Optional[PaperVMNDParams] = None,
        warm_start_file: Optional[str | Path] = None,
        log_to_console: bool = True,
    ) -> None:
        self.paper_params = params or PaperVMNDParams()
        self.warm_start_file = str(warm_start_file) if warm_start_file is not None else None
        self.warm_start_payload: Optional[Mapping[str, Any]] = None
        self.warm_start_summary: Dict[str, Any] = {}
        super().__init__(
            instance=instance,
            add_valid_inequalities=self.paper_params.use_valid_inequalities,
            log_to_console=log_to_console,
            numeric_focus=self.paper_params.numeric_focus,
            threads=self.paper_params.threads,
            seed=self.paper_params.seed,
            max_iterations=10**6,
            max_no_improve_rounds=10**6,
            global_polish_time=0.0,
            neighborhood_time=self.paper_params.mu_L,
            neighborhood_gap=self.paper_params.neighborhood_gap,
            improve_eps=self.paper_params.improve_eps,
        )

    def _build_warm_start_summary(self, vmnd_runtime_seconds: Optional[float] = None) -> Dict[str, Any]:
        payload = self.warm_start_payload if isinstance(self.warm_start_payload, Mapping) else {}
        meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
        diagnostics = payload.get("diagnostics", {}) if isinstance(payload, Mapping) else {}
        result_summary = payload.get("result_summary", {}) if isinstance(payload, Mapping) else {}

        lfrs_runtime = None
        if isinstance(result_summary, Mapping):
            for key in ("total_runtime_seconds", "runtime_seconds", "reduced_runtime_seconds"):
                if result_summary.get(key) is not None:
                    lfrs_runtime = float(result_summary[key])
                    break
        if lfrs_runtime is None and isinstance(diagnostics, Mapping):
            for key in ("total_runtime_seconds", "reduced_runtime_seconds"):
                if diagnostics.get(key) is not None:
                    lfrs_runtime = float(diagnostics[key])
                    break

        lfrs_cost = None
        if isinstance(result_summary, Mapping):
            for key in ("final_total_cost", "total_cost", "objective", "reduced_obj"):
                if result_summary.get(key) is not None:
                    lfrs_cost = float(result_summary[key])
                    break
        if lfrs_cost is None and isinstance(diagnostics, Mapping) and diagnostics.get("final_total_cost") is not None:
            lfrs_cost = float(diagnostics["final_total_cost"])
        if lfrs_cost is None and isinstance(diagnostics, Mapping) and diagnostics.get("reduced_obj") is not None:
            lfrs_cost = float(diagnostics["reduced_obj"])

        total_method_runtime = None
        if vmnd_runtime_seconds is not None:
            total_method_runtime = float(vmnd_runtime_seconds) + (lfrs_runtime or 0.0)

        return {
            "warm_start_path": self.warm_start_file,
            "meta": meta if isinstance(meta, Mapping) else {},
            "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else {},
            "result_summary": result_summary if isinstance(result_summary, Mapping) else {},
            "lfrs_runtime_seconds": lfrs_runtime,
            "lfrs_cost": lfrs_cost,
            "total_method_runtime_seconds": total_method_runtime,
        }

    def _all_routes(self, incumbent, d) -> List[Tuple[int, List[int]]]:
        routes_map = incumbent.get("period_routes") or self._route_reconstruction(incumbent["x"], d)
        flat: List[Tuple[int, List[int]]] = []
        for t in d["periods"]:
            for route in routes_map.get(t, []):
                flat.append((t, route))
        return flat

    def _route_load(self, route: List[int], incumbent, t: int) -> float:
        return sum(incumbent["q"].get((node, t), 0.0) for node in route[1:-1])

    def _route_unused_capacity_miles(self, route: List[int], incumbent, d, t: int) -> float:
        load = self._route_load(route, incumbent, t)
        if len(route) < 2:
            return 0.0
        distance = 0.0
        for a, b in zip(route[:-1], route[1:]):
            distance += d["dist_nodes"].get((a, b), 0.0)
        return max(0.0, self.instance.vehicle_capacity - load) * distance

    def _dynamic_gamma(self, elapsed_ratio: float) -> float:
        p = self.paper_params
        return p.gamma_max - (p.gamma_max - p.gamma_min) * min(max(elapsed_ratio, 0.0), 1.0)

    def _dynamic_bcp_time(self, elapsed_ratio: float) -> float:
        p = self.paper_params
        return p.mu_B_min + (p.mu_B_max - p.mu_B_min) * min(max(elapsed_ratio, 0.0), 1.0)

    def _warm_start_to_incumbent(self, d: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Dict[Any, float]]:
        """Normalize an external warm-start payload to this solver's internal key types.

        The instance model uses string module ids (e.g. "1", "2"), while warm-start files
        exported by LFRS often store module ids as integers. If we copy z-keys directly,
        the incumbent keeps parallel entries like (3, 1) instead of updating the expected
        internal keys (3, "1"). That makes fixed opening costs disappear from objective
        calculations and causes open_pls to look empty even though the warm-start is valid.
        """
        inc = self._empty_incumbent(d)

        if "z" in raw:
            for (i, m), value in raw["z"].items():
                inc["z"][(int(i), str(m))] = float(value)
        for key in ("y", "v", "q", "I", "x", "w"):
            if key in raw:
                inc[key].update(raw[key])
        inc["u"][None] = raw.get("u", 0)
        inc["period_routes"] = raw.get("period_routes", {})
        if not inc["period_routes"]:
            inc["period_routes"] = self._route_reconstruction(inc["x"], d)
        return inc

    def _incumbent_to_warm_payload(self, incumbent, d) -> Dict[str, Any]:
        return {
            "z": {f"{i},{m}": int(round(v)) for (i, m), v in incumbent["z"].items() if abs(v) > 1e-9},
            "y": {f"{i},{k}": int(round(v)) for (i, k), v in incumbent["y"].items() if abs(v) > 1e-9},
            "v": {f"{i},{t}": int(round(vv)) for (i, t), vv in incumbent["v"].items() if abs(vv) > 1e-9},
            "q": {f"{i},{t}": float(vv) for (i, t), vv in incumbent["q"].items() if abs(vv) > 1e-9},
            "I": {f"{i},{t}": float(vv) for (i, t), vv in incumbent["I"].items() if abs(vv) > 1e-9},
            "x": {f"{i},{j},{t}": int(round(vv)) for (i, j, t), vv in incumbent["x"].items() if abs(vv) > 1e-9},
            "w": {f"{i},{t}": float(vv) for (i, t), vv in incumbent["w"].items() if abs(vv) > 1e-9},
            "u": int(round(incumbent["u"][None])),
            "period_routes": {str(t): routes for t, routes in incumbent.get("period_routes", {}).items()},
        }

    def _free_theta1(self, vars_dict, incumbent, d, selected_period: int) -> None:
        t = selected_period
        for i in d["pl_ids"]:
            self._set_free(vars_dict["v"][i, t], 0.0, 1.0)
            self._set_free(vars_dict["q"][i, t], 0.0, GRB.INFINITY)
            self._set_free(vars_dict["w"][i, t], 0.0, GRB.INFINITY)
            self._set_free(vars_dict["I"][i, t], 0.0, GRB.INFINITY)
        for (i, j, tt) in d["arcs"]:
            if tt == t:
                self._set_free(vars_dict["x"][i, j, tt], 0.0, 1.0)
        self._set_free(vars_dict["u"], 0.0, max(1, int(round(incumbent["u"][None])) + 1))

    def _select_routes_for_theta(self, incumbent, d, theta: int) -> List[Tuple[int, List[int]]]:
        routes = self._all_routes(incumbent, d)
        if not routes:
            return []
        if theta == 2:
            scored = sorted(routes, key=lambda tr: self._route_load(tr[1], incumbent, tr[0]))
            portion = self.paper_params.xi2
        elif theta == 3:
            scored = sorted(routes, key=lambda tr: self._route_unused_capacity_miles(tr[1], incumbent, d, tr[0]), reverse=True)
            portion = self.paper_params.xi3
        elif theta == 4:
            scored = sorted(routes, key=lambda tr: self._route_unused_capacity_miles(tr[1], incumbent, d, tr[0]), reverse=True)
            portion = self.paper_params.xi4
        elif theta == 5:
            scored = list(routes)
            self.rng.shuffle(scored)
            portion = self.paper_params.xi5
        else:
            return []
        count = max(1, int(math.ceil(portion * len(routes))))
        return scored[:count]

    def _free_theta2(self, vars_dict, incumbent, d, selected_routes: List[Tuple[int, List[int]]]) -> None:
        pls = {node for _, route in selected_routes for node in route[1:-1]}
        self._free_pl_neighborhood(vars_dict, incumbent, d, list(pls))

    def _free_theta3(self, vars_dict, incumbent, d, selected_routes: List[Tuple[int, List[int]]]) -> None:
        pls = {node for _, route in selected_routes for node in route[1:-1]}
        affected_srs = [k for k, i in self._current_assignment(incumbent, d).items() if i in pls]
        self._free_assignment_neighborhood(vars_dict, incumbent, d, affected_srs)

    def _free_theta4(self, vars_dict, incumbent, d, selected_routes: List[Tuple[int, List[int]]]) -> None:
        pls = {node for _, route in selected_routes for node in route[1:-1]}
        self._free_pl_neighborhood(vars_dict, incumbent, d, list(pls))

    def _free_theta5(self, vars_dict, incumbent, d, selected_routes: List[Tuple[int, List[int]]]) -> None:
        current_open = self._current_open_pls(incumbent, d)
        pls = {node for _, route in selected_routes for node in route[1:-1]}
        expanded = set(pls)
        for i in list(pls):
            closed_neighbors = [j for j in d["pl_ids"] if j not in current_open and j != i]
            if closed_neighbors:
                jstar = min(closed_neighbors, key=lambda j: d["dist_nodes"][(i, j)])
                expanded.add(jstar)
        self._free_pl_neighborhood(vars_dict, incumbent, d, list(expanded))
        for i in d["pl_ids"]:
            for k in d["sr_ids"]:
                self._set_free(vars_dict["y"][i, k], 0.0, 1.0)

    def solve(self) -> Dict[str, Any]:
        d = self.builder._prepare_data()
        if self.warm_start_file:
            self.warm_start_payload = load_warm_start_payload(self.warm_start_file)
            raw_warm = load_warm_start_file(self.warm_start_file, payload=self.warm_start_payload)
            incumbent = self._warm_start_to_incumbent(d, raw_warm)
            best_sol = self._incumbent_to_solution_dict(incumbent, d)
            best_sol["initial_solution_type"] = "external_warm_start_file"
            best_sol["warm_start_file"] = self.warm_start_file
            best_sol["warm_start_input_summary"] = self._build_warm_start_summary()
        else:
            incumbent, best_sol = self._heuristic_initial_solution()

        best_obj = self._compute_objective(incumbent, d)
        if self.log_to_console:
            print(f"[VMND] Initial incumbent = {best_obj:.4f}")

        start_time = time.time()
        iteration = 0
        log_rows: List[Dict[str, Any]] = []
        stagnant_bcp_rounds = 0
        termination_reason = "time_limit_reached"

        while time.time() - start_time < self.paper_params.mu_V:
            iteration += 1
            elapsed = time.time() - start_time
            remaining_total = self.paper_params.mu_V - elapsed
            if remaining_total <= 1.0:
                termination_reason = "global_time_exhausted"
                break

            elapsed_ratio = elapsed / max(self.paper_params.mu_V, 1e-9)
            gamma_now = self._dynamic_gamma(elapsed_ratio)
            bcp_time = min(self._dynamic_bcp_time(elapsed_ratio), remaining_total)
            if bcp_time <= 1.0:
                termination_reason = "global_time_exhausted"
                break

            model, vars_dict, d = self.builder.build_model(
                time_limit=bcp_time,
                mip_gap=self.paper_params.neighborhood_gap,
                name_suffix=f"_BCP_{iteration}",
            )
            self._apply_start(vars_dict, incumbent, d)
            model.optimize()

            rel_improvement = 0.0
            bcp_improved = False
            cand_obj = best_obj
            run_lsp = False

            if model.SolCount > 0:
                candidate = self._extract_incumbent(model, vars_dict, d)
                cand_obj = self._compute_objective(candidate, d)
                if cand_obj + 1e-9 < best_obj:
                    rel_improvement = (best_obj - cand_obj) / max(abs(best_obj), 1e-9)
                    incumbent = candidate
                    best_obj = cand_obj
                    best_sol = self._incumbent_to_solution_dict(candidate, d, runtime=model.Runtime)
                    bcp_improved = True
                    stagnant_bcp_rounds = 0
                    log_rows.append(
                        {
                            "phase": "BCP",
                            "iteration": iteration,
                            "objective": best_obj,
                            "relative_improvement": rel_improvement,
                            "runtime": float(model.Runtime),
                            "best_bound": float(model.ObjBound) if hasattr(model, "ObjBound") else None,
                            "mip_gap": float(model.MIPGap) if hasattr(model, "MIPGap") else None,
                        }
                    )
                    if self.log_to_console:
                        print(
                            f"[VMND] Iter {iteration} BCP improved to {best_obj:.4f}; "
                            f"rel={rel_improvement:.5f}; gamma={gamma_now:.5f}"
                        )
                    run_lsp = rel_improvement >= gamma_now

            if not bcp_improved:
                stagnant_bcp_rounds += 1
                gap_now = float(model.MIPGap) if hasattr(model, "MIPGap") else None
                best_bound = float(model.ObjBound) if hasattr(model, "ObjBound") else None
                log_rows.append(
                    {
                        "phase": "BCP_no_improve",
                        "iteration": iteration,
                        "objective": best_obj,
                        "candidate_objective": cand_obj,
                        "runtime": float(model.Runtime),
                        "best_bound": best_bound,
                        "mip_gap": gap_now,
                        "stagnant_bcp_rounds": stagnant_bcp_rounds,
                    }
                )
                if self.log_to_console:
                    print(
                        f"[VMND] Iter {iteration} BCP no improvement; incumbent={best_obj:.4f}, "
                        f"candidate={cand_obj:.4f}, stagnant={stagnant_bcp_rounds}"
                    )

                solved_to_requested_gap = False
                try:
                    solved_to_requested_gap = gap_now is not None and gap_now <= self.paper_params.neighborhood_gap + 1e-9
                except Exception:
                    solved_to_requested_gap = False

                if (
                    self.paper_params.stop_if_bcp_repeats_incumbent
                    and model.SolCount > 0
                    and abs(cand_obj - best_obj) <= self.paper_params.improve_eps
                    and solved_to_requested_gap
                ):
                    termination_reason = "bcp_repeated_same_incumbent_at_requested_gap"
                    if self.log_to_console:
                        print("[VMND] Stop early: BCP rebuilt the same incumbent and already met the requested gap.")
                    break

                if stagnant_bcp_rounds >= self.paper_params.max_stagnant_bcp_rounds:
                    termination_reason = f"stagnant_bcp_rounds_{stagnant_bcp_rounds}"
                    if self.log_to_console:
                        print(f"[VMND] Stop early: {stagnant_bcp_rounds} consecutive BCP rounds produced no improvement.")
                    break

                run_lsp = self.paper_params.force_lsp_after_bcp_no_improve
                if not run_lsp:
                    if self.log_to_console:
                        print(f"[VMND] Iter {iteration} skip LSP (rel={rel_improvement:.5f}, gamma={gamma_now:.5f})")
                    continue
                if self.log_to_console:
                    print(
                        f"[VMND] Iter {iteration} trigger LSP after BCP stagnation "
                        f"(stagnant={stagnant_bcp_rounds}, gamma={gamma_now:.5f})"
                    )

            if not run_lsp and rel_improvement < gamma_now:
                if self.log_to_console:
                    print(f"[VMND] Iter {iteration} skip LSP (rel={rel_improvement:.5f}, gamma={gamma_now:.5f})")
                continue

            improved_in_lsp = True
            while improved_in_lsp and time.time() - start_time < self.paper_params.mu_V:
                improved_in_lsp = False
                for theta in self.paper_params.operator_order:
                    remaining_total = self.paper_params.mu_V - (time.time() - start_time)
                    if remaining_total <= 1.0:
                        termination_reason = "global_time_exhausted"
                        break

                    original_neighborhood_time = self.neighborhood_time
                    self.neighborhood_time = min(original_neighborhood_time, remaining_total)
                    try:
                        if theta == 1:
                            scored_periods = sorted(
                                [(sum(incumbent["q"][(i, t)] for i in d["pl_ids"]), t) for t in d["periods"]],
                                reverse=True,
                            )
                            selected_period = scored_periods[0][1] if scored_periods else d["periods"][0]
                            result = self._solve_submip(incumbent, self._free_theta1, "theta1", (selected_period,))
                        else:
                            selected_routes = self._select_routes_for_theta(incumbent, d, theta)
                            if not selected_routes:
                                continue
                            callback = {2: self._free_theta2, 3: self._free_theta3, 4: self._free_theta4, 5: self._free_theta5}[theta]
                            result = self._solve_submip(incumbent, callback, f"theta{theta}", (selected_routes,))
                    finally:
                        self.neighborhood_time = original_neighborhood_time

                    if result is None:
                        continue
                    new_inc, new_sol = result
                    new_obj = self._compute_objective(new_inc, d)
                    if new_obj + 1e-9 < best_obj:
                        incumbent = new_inc
                        best_obj = new_obj
                        best_sol = new_sol
                        stagnant_bcp_rounds = 0
                        log_rows.append({"phase": f"LSP_theta{theta}", "iteration": iteration, "objective": best_obj})
                        improved_in_lsp = True
                        if self.log_to_console:
                            print(f"[VMND] Iter {iteration} theta{theta} improved to {best_obj:.4f}")
                        break

        best_sol["algorithm"] = "VMND"
        best_sol["paper_params"] = self.paper_params.to_dict()
        best_sol["vmnd_iterations"] = iteration
        best_sol["vmnd_improvement_log"] = log_rows
        best_sol["runtime_seconds_total"] = time.time() - start_time
        best_sol["termination_reason"] = termination_reason
        best_sol["warm_start"] = self._incumbent_to_warm_payload(incumbent, d)
        if self.warm_start_file:
            best_sol["warm_start_input_summary"] = self._build_warm_start_summary(best_sol["runtime_seconds_total"])
            if best_sol["warm_start_input_summary"].get("total_method_runtime_seconds") is not None:
                best_sol["total_method_runtime_seconds"] = best_sol["warm_start_input_summary"]["total_method_runtime_seconds"]
        else:
            best_sol["total_method_runtime_seconds"] = best_sol["runtime_seconds_total"]
        return best_sol


# ============================================================
# CLI
# ============================================================


def write_default_params(path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(PaperVMNDParams().to_dict(), f, indent=2)


def override_params_time_budget(
    params: PaperVMNDParams,
    *,
    time_limit_seconds: Optional[float] = None,
    time_limit_minutes: Optional[float] = None,
) -> PaperVMNDParams:
    total_seconds = None
    if time_limit_seconds is not None:
        total_seconds = float(time_limit_seconds)
    elif time_limit_minutes is not None:
        total_seconds = float(time_limit_minutes) * 60.0

    if total_seconds is None:
        return params
    if total_seconds <= 0:
        raise ValueError("Time limit must be positive.")

    scale = total_seconds / 10800.0
    params.mu_V = float(total_seconds)
    params.mu_B_min = max(5.0, 120.0 * scale)
    params.mu_B_max = max(params.mu_B_min, 360.0 * scale)
    params.mu_L = max(5.0, 300.0 * scale)
    return params



# ============================================================
# Aggressive paper-style VMND variant
# ============================================================

class PLLRPVMNDPaperAggressive(PLLRPVMNDPaper):
    """More aggressive search around an external LFRS incumbent.

    This variant keeps exact acceptance (only strictly improving incumbents are accepted),
    but it expands neighborhoods and adds a diversification sub-MIP when standard theta
    neighborhoods fail repeatedly.
    """

    def _comp_penalty_by_sr(self, incumbent, d) -> Dict[int, float]:
        penalties: Dict[int, float] = {}
        assign = self._current_assignment(incumbent, d)
        cc = self.instance.compensation_cost
        for k in d["sr_ids"]:
            i = assign.get(k)
            if i is None:
                penalties[k] = 0.0
                continue
            dist = d["dist_assign"].get((i, k), 0.0)
            penalties[k] = cc * dist * sum(d["demand"][(k, t)] for t in d["periods"])
        return penalties

    def _select_diversify_srs(self, incumbent, d) -> List[int]:
        frac = float(getattr(self.paper_params, "diversify_sr_fraction", 0.10))
        penalties = self._comp_penalty_by_sr(incumbent, d)
        ranked = sorted(d["sr_ids"], key=lambda k: penalties.get(k, 0.0), reverse=True)
        if not ranked:
            return []
        count = max(4, min(len(ranked), int(math.ceil(frac * len(ranked)))))
        head = ranked[:max(count, 8)]
        self.rng.shuffle(head)
        picked = sorted(head[:count], key=lambda k: penalties.get(k, 0.0), reverse=True)
        return picked

    def _nearest_closed_pls(self, incumbent, d, sr_subset: Sequence[int]) -> List[int]:
        extra = int(getattr(self.paper_params, "diversify_extra_closed_pls", 2))
        if extra <= 0:
            return []
        open_pls = set(self._current_open_pls(incumbent, d))
        chosen: List[int] = []
        used = set()
        for k in sr_subset:
            candidates = sorted(
                (i for i in d["pl_ids"] if i not in open_pls and i not in used),
                key=lambda i: d["dist_assign"].get((i, k), float("inf")),
            )
            for i in candidates[:extra]:
                if i not in used:
                    used.add(i)
                    chosen.append(i)
        return chosen

    def _select_diversify_periods(self, incumbent, d, sr_subset: Sequence[int]) -> List[int]:
        frac = float(getattr(self.paper_params, "diversify_time_fraction", 0.50))
        scores = []
        for t in d["periods"]:
            total = sum(d["demand"][(k, t)] for k in sr_subset)
            scores.append((total, t))
        scores.sort(reverse=True)
        count = max(1, min(len(scores), int(math.ceil(frac * len(scores)))))
        return [t for _, t in scores[:count]]

    def _free_diversify(
        self,
        vars_dict,
        incumbent,
        d,
        sr_subset: Sequence[int],
        extra_pls: Sequence[int],
        periods_subset: Sequence[int],
    ) -> None:
        current_assign = self._current_assignment(incumbent, d)
        anchor_pls = {current_assign[k] for k in sr_subset if k in current_assign}
        pl_set = set(anchor_pls) | set(extra_pls)
        tset = set(periods_subset)

        # Reassign the selected SRs freely.
        for k in sr_subset:
            for i in d["pl_ids"]:
                self._set_free(vars_dict["y"][i, k], 0.0, 1.0)

        # Free opening / replenishment decisions for anchor and promising nearby closed PLs.
        for i in pl_set:
            for m in d["module_ids"]:
                self._set_free(vars_dict["z"][i, m], 0.0, 1.0)
            for t in d["periods"]:
                self._set_free(vars_dict["v"][i, t], 0.0, 1.0)
                self._set_free(vars_dict["q"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["w"][i, t], 0.0, GRB.INFINITY)
                self._set_free(vars_dict["I"][i, t], 0.0, GRB.INFINITY)

        # Free routing only for affected periods, but on all arcs incident to selected PLs.
        for (i, j, t) in d["arcs"]:
            if t in tset and (i in pl_set or j in pl_set):
                self._set_free(vars_dict["x"][i, j, t], 0.0, 1.0)

        # Relax fleet a little more than standard neighborhoods.
        max_v = max(1, int(round(incumbent["u"][None])) + max(2, len(pl_set)))
        self._set_free(vars_dict["u"], 0.0, max_v)

    def solve(self) -> Dict[str, Any]:
        d = self.builder._prepare_data()

        if self.warm_start_file:
            self.warm_start_payload = load_warm_start_payload(self.warm_start_file)
            raw_warm = load_warm_start_file(self.warm_start_file, payload=self.warm_start_payload)
            incumbent = self._warm_start_to_incumbent(d, raw_warm)
            best_sol = self._incumbent_to_solution_dict(incumbent, d, runtime=0.0)
            best_sol["initial_solution_type"] = "external_warm_start_file"
            best_sol["warm_start_file"] = self.warm_start_file
            best_sol["warm_start_input_summary"] = self._build_warm_start_summary()
        else:
            incumbent, best_sol = self._heuristic_initial_solution()

        best_obj = self._compute_objective(incumbent, d)
        if self.log_to_console:
            print(f"[VMND-AGG] Initial incumbent = {best_obj:.4f}")

        start_time = time.time()
        iteration = 0
        log_rows: List[Dict[str, Any]] = []
        stagnant_bcp_rounds = 0
        termination_reason = "time_limit_reached"

        while time.time() - start_time < self.paper_params.mu_V:
            iteration += 1
            elapsed = time.time() - start_time
            remaining_total = self.paper_params.mu_V - elapsed
            if remaining_total <= 1.0:
                termination_reason = "global_time_exhausted"
                break

            elapsed_ratio = elapsed / max(self.paper_params.mu_V, 1e-9)
            gamma_now = self._dynamic_gamma(elapsed_ratio)
            bcp_time = min(self._dynamic_bcp_time(elapsed_ratio), remaining_total)
            if bcp_time <= 1.0:
                termination_reason = "global_time_exhausted"
                break

            model, vars_dict, d = self.builder.build_model(
                time_limit=bcp_time,
                mip_gap=self.paper_params.neighborhood_gap,
                name_suffix=f"_BCP_{iteration}",
            )
            self._apply_start(vars_dict, incumbent, d)
            model.optimize()

            rel_improvement = 0.0
            bcp_improved = False
            cand_obj = best_obj
            run_lsp = False

            if model.SolCount > 0:
                candidate = self._extract_incumbent(model, vars_dict, d)
                cand_obj = self._compute_objective(candidate, d)
                if cand_obj + 1e-9 < best_obj:
                    rel_improvement = (best_obj - cand_obj) / max(abs(best_obj), 1e-9)
                    incumbent = candidate
                    best_obj = cand_obj
                    best_sol = self._incumbent_to_solution_dict(candidate, d, runtime=model.Runtime)
                    bcp_improved = True
                    stagnant_bcp_rounds = 0
                    log_rows.append(
                        {
                            "phase": "BCP",
                            "iteration": iteration,
                            "objective": best_obj,
                            "relative_improvement": rel_improvement,
                            "runtime": float(model.Runtime),
                            "best_bound": float(model.ObjBound) if hasattr(model, "ObjBound") else None,
                            "mip_gap": float(model.MIPGap) if hasattr(model, "MIPGap") else None,
                        }
                    )
                    if self.log_to_console:
                        print(
                            f"[VMND-AGG] Iter {iteration} BCP improved to {best_obj:.4f}; "
                            f"rel={rel_improvement:.5f}; gamma={gamma_now:.5f}"
                        )
                    run_lsp = rel_improvement >= gamma_now

            if not bcp_improved:
                stagnant_bcp_rounds += 1
                gap_now = float(model.MIPGap) if hasattr(model, "MIPGap") else None
                best_bound = float(model.ObjBound) if hasattr(model, "ObjBound") else None
                log_rows.append(
                    {
                        "phase": "BCP_no_improve",
                        "iteration": iteration,
                        "objective": best_obj,
                        "candidate_objective": cand_obj,
                        "runtime": float(model.Runtime),
                        "best_bound": best_bound,
                        "mip_gap": gap_now,
                        "stagnant_bcp_rounds": stagnant_bcp_rounds,
                    }
                )
                if self.log_to_console:
                    print(
                        f"[VMND-AGG] Iter {iteration} BCP no improvement; incumbent={best_obj:.4f}, "
                        f"candidate={cand_obj:.4f}, stagnant={stagnant_bcp_rounds}"
                    )
                run_lsp = getattr(self.paper_params, "force_lsp_after_bcp_no_improve", True)

            if not run_lsp and rel_improvement < gamma_now:
                if self.log_to_console:
                    print(f"[VMND-AGG] Iter {iteration} skip LSP (rel={rel_improvement:.5f}, gamma={gamma_now:.5f})")
                continue

            improved_in_lsp = False

            # Two full passes through the theta neighborhoods before diversification.
            lsp_passes = int(getattr(self.paper_params, "lsp_rounds_per_iteration", 2))
            for pass_idx in range(lsp_passes):
                if time.time() - start_time >= self.paper_params.mu_V:
                    termination_reason = "global_time_exhausted"
                    break
                for theta in self.paper_params.operator_order:
                    remaining_total = self.paper_params.mu_V - (time.time() - start_time)
                    if remaining_total <= 1.0:
                        termination_reason = "global_time_exhausted"
                        break

                    original_neighborhood_time = self.neighborhood_time
                    self.neighborhood_time = min(original_neighborhood_time, remaining_total)
                    result = None
                    try:
                        if theta == 1:
                            scored_periods = sorted(
                                [(sum(incumbent["q"][(i, t)] for i in d["pl_ids"]), t) for t in d["periods"]],
                                reverse=True,
                            )
                            selected_period = scored_periods[0][1] if scored_periods else d["periods"][0]
                            result = self._solve_submip(incumbent, self._free_theta1, "theta1", (selected_period,))
                        else:
                            selected_routes = self._select_routes_for_theta(incumbent, d, theta)
                            if selected_routes:
                                callback = {
                                    2: self._free_theta2,
                                    3: self._free_theta3,
                                    4: self._free_theta4,
                                    5: self._free_theta5,
                                }[theta]
                                result = self._solve_submip(incumbent, callback, f"theta{theta}", (selected_routes,))
                    finally:
                        self.neighborhood_time = original_neighborhood_time

                    if result is None:
                        log_rows.append(
                            {
                                "phase": f"LSP_theta{theta}_no_improve",
                                "iteration": iteration,
                                "pass": pass_idx + 1,
                                "objective": best_obj,
                            }
                        )
                        continue

                    new_inc, new_sol = result
                    new_obj = self._compute_objective(new_inc, d)
                    if new_obj + 1e-9 < best_obj:
                        incumbent = new_inc
                        best_obj = new_obj
                        best_sol = new_sol
                        stagnant_bcp_rounds = 0
                        improved_in_lsp = True
                        log_rows.append(
                            {
                                "phase": f"LSP_theta{theta}_improve",
                                "iteration": iteration,
                                "pass": pass_idx + 1,
                                "objective": best_obj,
                            }
                        )
                        if self.log_to_console:
                            print(f"[VMND-AGG] Iter {iteration} theta{theta} improved to {best_obj:.4f}")
                        break
                if improved_in_lsp or termination_reason == "global_time_exhausted":
                    break

            # Diversification phase: free top-penalty SRs + nearby closed PLs + busiest affected periods.
            if (not improved_in_lsp) and (time.time() - start_time < self.paper_params.mu_V):
                sr_subset = self._select_diversify_srs(incumbent, d)
                if sr_subset:
                    extra_pls = self._nearest_closed_pls(incumbent, d, sr_subset)
                    periods_subset = self._select_diversify_periods(incumbent, d, sr_subset)
                    remaining_total = self.paper_params.mu_V - (time.time() - start_time)
                    original_neighborhood_time = self.neighborhood_time
                    budget_share = float(getattr(self.paper_params, "diversify_budget_share", 0.60))
                    self.neighborhood_time = min(max(original_neighborhood_time, 120.0), remaining_total * budget_share)
                    result = None
                    try:
                        result = self._solve_submip(
                            incumbent,
                            self._free_diversify,
                            "diversify",
                            (sr_subset, extra_pls, periods_subset),
                        )
                    finally:
                        self.neighborhood_time = original_neighborhood_time

                    if result is None:
                        log_rows.append(
                            {
                                "phase": "LSP_diversify_no_improve",
                                "iteration": iteration,
                                "objective": best_obj,
                                "selected_srs": list(sr_subset),
                                "extra_pls": list(extra_pls),
                                "periods": list(periods_subset),
                            }
                        )
                    else:
                        new_inc, new_sol = result
                        new_obj = self._compute_objective(new_inc, d)
                        if new_obj + 1e-9 < best_obj:
                            incumbent = new_inc
                            best_obj = new_obj
                            best_sol = new_sol
                            stagnant_bcp_rounds = 0
                            improved_in_lsp = True
                            log_rows.append(
                                {
                                    "phase": "LSP_diversify_improve",
                                    "iteration": iteration,
                                    "objective": best_obj,
                                    "selected_srs": list(sr_subset),
                                    "extra_pls": list(extra_pls),
                                    "periods": list(periods_subset),
                                }
                            )
                            if self.log_to_console:
                                print(f"[VMND-AGG] Iter {iteration} diversify improved to {best_obj:.4f}")

        best_sol["algorithm"] = "VMND_AGGRESSIVE"
        best_sol["paper_params"] = self.paper_params.to_dict()
        best_sol["vmnd_iterations"] = iteration
        best_sol["vmnd_improvement_log"] = log_rows
        best_sol["runtime_seconds_total"] = time.time() - start_time
        best_sol["termination_reason"] = termination_reason
        best_sol["warm_start"] = self._incumbent_to_warm_payload(incumbent, d)
        if self.warm_start_file:
            best_sol["warm_start_input_summary"] = self._build_warm_start_summary(best_sol["runtime_seconds_total"])
            if best_sol["warm_start_input_summary"].get("total_method_runtime_seconds") is not None:
                best_sol["total_method_runtime_seconds"] = best_sol["warm_start_input_summary"]["total_method_runtime_seconds"]
        else:
            best_sol["total_method_runtime_seconds"] = best_sol["runtime_seconds_total"]
        return best_sol



def solve_instance_file(
    instance_path: str | Path,
    output_path: Optional[str | Path] = None,
    warm_start_file: Optional[str | Path] = None,
    params_file: Optional[str | Path] = None,
    quiet: bool = False,
    time_limit_seconds: Optional[float] = None,
    time_limit_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    instance = load_instance_any(instance_path)
    params = PaperVMNDParams()
    if params_file is not None:
        with open(params_file, "r", encoding="utf-8") as f:
            params = PaperVMNDParams.from_dict(json.load(f))
    params = override_params_time_budget(
        params,
        time_limit_seconds=time_limit_seconds,
        time_limit_minutes=time_limit_minutes,
    )
    solver = PLLRPVMNDPaperAggressive(instance=instance, params=params, warm_start_file=warm_start_file, log_to_console=not quiet)
    solution = solver.solve()
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(solution), f, indent=2)
    return solution


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paper-parameter VMND for PL-LRP with external LFRS warm-start file support.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate a random PL-LRP instance using paper defaults.")
    g.add_argument("--output", required=True)
    g.add_argument("--n-pl", type=int, required=True)
    g.add_argument("--n-sr", type=int, required=True)
    g.add_argument("--n-modules", type=int, required=True)
    g.add_argument("--n-periods", type=int, required=True)
    g.add_argument("--pattern", choices=["uniform", "bimodal"], default="uniform")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--name", default=None)

    p = sub.add_parser("write-default-params", help="Write the paper VMND parameter file to JSON.")
    p.add_argument("--output", required=True)

    s = sub.add_parser("solve", help="Solve with VMND. Supports warm-start output files from LFRS or previous runs.")
    s.add_argument("--instance", required=True, help="JSON instance file, compact VMND format or LFRS plain format.")
    s.add_argument("--output", required=False, help="Optional JSON output path.")
    s.add_argument("--warm-start-file", required=False, help="Optional JSON warm-start file produced by LFRS/VMND/B&C.")
    s.add_argument("--params-file", required=False, help="Optional JSON overriding the paper parameters.")
    s.add_argument("--time-limit-seconds", type=float, required=False, help="Optional total VMND time budget in seconds.")
    s.add_argument("--time-limit-minutes", type=float, required=False, help="Optional total VMND time budget in minutes.")
    s.add_argument("--quiet", action="store_true")

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

    if args.command == "write-default-params":
        write_default_params(args.output)
        print(f"Wrote default paper parameters to {args.output}")
        return

    if args.command == "solve":
        solution = solve_instance_file(
            instance_path=args.instance,
            output_path=args.output,
            warm_start_file=args.warm_start_file,
            params_file=args.params_file,
            quiet=args.quiet,
            time_limit_seconds=args.time_limit_seconds,
            time_limit_minutes=args.time_limit_minutes,
        )
        print(json.dumps(_json_safe(solution), indent=2))
        return


if __name__ == "__main__":
    cli()
