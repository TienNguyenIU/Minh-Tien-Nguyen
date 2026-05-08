from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
# =========================
# GUROBI WLS INLINE LICENSE
# =========================
# DÁN THÔNG TIN LICENSE CỦA BẠN VÀO 3 DÒNG BÊN DƯỚI.
# Nếu để trống/0, LFRS sẽ fallback về cách tạo model Gurobi như cũ.
WLS_PARAMS = {
     "WLSACCESSID": "d019c5a0-148e-4e31-9780-6e222500367b",
    "WLSSECRET": "4c6f3287-9593-428b-a2da-23934b32bb4e",
    "LICENSEID": 2804261,
}


Arc = Tuple[int, int]
ZKey = Tuple[int, int]
YKey = Tuple[int, int]
VTKey = Tuple[int, int]
QTKey = Tuple[int, int]
ITKey = Tuple[int, int]
XKey = Tuple[int, int, int]
WTKey = Tuple[int, int]


@dataclass
class PLLRPInstance:
    """Input data for the PL-LRP warm start."""

    pl_ids: Sequence[int]
    sr_ids: Sequence[int]
    periods: Sequence[int]
    module_ids: Sequence[int]
    module_capacity: Mapping[int, float]
    opening_cost: Mapping[ZKey, float]
    demand: Mapping[Tuple[int, int], float]
    coords: Mapping[int, Tuple[float, float]]
    vehicle_capacity: float
    alpha: float
    beta: float
    pl_coords: Optional[Mapping[int, Tuple[float, float]]] = None
    sr_coords: Optional[Mapping[int, Tuple[float, float]]] = None
    vehicle_fixed_cost: float = 0.0
    dist_first_echelon: Optional[Mapping[Arc, float]] = None
    dist_pl_sr: Optional[Mapping[Tuple[int, int], float]] = None
    depot_id: int = 0

    def __post_init__(self) -> None:
        self.pl_ids = list(self.pl_ids)
        self.sr_ids = list(self.sr_ids)
        self.periods = list(self.periods)
        self.module_ids = list(self.module_ids)

        coords_map = dict(self.coords)
        if self.depot_id not in coords_map:
            raise ValueError("coords must contain the depot id")

        pl_map = dict(self.pl_coords) if self.pl_coords is not None else {}
        sr_map = dict(self.sr_coords) if self.sr_coords is not None else {}

        for i in self.pl_ids:
            if i in pl_map:
                continue
            if i not in coords_map:
                raise ValueError(f"Missing coordinate for PL {i}")
            pl_map[i] = coords_map[i]

        for k in self.sr_ids:
            if k in sr_map:
                continue
            if k not in coords_map:
                raise ValueError(f"Missing coordinate for SR {k}")
            sr_map[k] = coords_map[k]

        self.coords = coords_map
        self.pl_coords = pl_map
        self.sr_coords = sr_map

        for m in self.module_ids:
            if m not in self.module_capacity:
                raise ValueError(f"Missing capacity for module {m}")
        for i in self.pl_ids:
            for m in self.module_ids:
                if (i, m) not in self.opening_cost:
                    raise ValueError(f"Missing opening cost for {(i, m)}")
        for k in self.sr_ids:
            for t in self.periods:
                if (k, t) not in self.demand:
                    raise ValueError(f"Missing demand for {(k, t)}")

    def d_first(self, i: int, j: int) -> float:
        if self.dist_first_echelon is not None and (i, j) in self.dist_first_echelon:
            return float(self.dist_first_echelon[(i, j)])

        if i == self.depot_id:
            xi, yi = self.coords[self.depot_id]
        else:
            if i not in self.pl_coords:
                raise ValueError(f"Missing coordinate for PL {i}")
            xi, yi = self.pl_coords[i]

        if j == self.depot_id:
            xj, yj = self.coords[self.depot_id]
        else:
            if j not in self.pl_coords:
                raise ValueError(f"Missing coordinate for PL {j}")
            xj, yj = self.pl_coords[j]

        return ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5

    def d_pl_sr(self, i: int, k: int) -> float:
        if self.dist_pl_sr is not None and (i, k) in self.dist_pl_sr:
            return float(self.dist_pl_sr[(i, k)])
        if i not in self.pl_coords:
            raise ValueError(f"Missing coordinate for PL {i}")
        if k not in self.sr_coords:
            raise ValueError(
                f"Missing coordinate for SR {k}; either provide it in sr_coords/coords or pass dist_pl_sr"
            )
        xi, yi = self.pl_coords[i]
        xk, yk = self.sr_coords[k]
        return ((xi - xk) ** 2 + (yi - yk) ** 2) ** 0.5


@dataclass
class ReducedSolution:
    z: Dict[ZKey, int]
    y: Dict[YKey, int]
    v: Dict[VTKey, int]
    q: Dict[QTKey, float]
    I: Dict[ITKey, float]
    objective_value: float
    status: str
    runtime: float
    hat_R: Dict[int, float]


@dataclass
class WarmStartSolution:
    z: Dict[ZKey, int] = field(default_factory=dict)
    y: Dict[YKey, int] = field(default_factory=dict)
    v: Dict[VTKey, int] = field(default_factory=dict)
    q: Dict[QTKey, float] = field(default_factory=dict)
    I: Dict[ITKey, float] = field(default_factory=dict)
    x: Dict[XKey, int] = field(default_factory=dict)
    w: Dict[WTKey, float] = field(default_factory=dict)
    u: int = 0
    period_routes: Dict[int, List[List[int]]] = field(default_factory=dict)
    reduced_obj: float = 0.0
    note: str = ""

    def to_start_dict(self) -> Dict[str, Dict]:
        return {
            "z": self.z,
            "y": self.y,
            "v": self.v,
            "q": self.q,
            "I": self.I,
            "x": self.x,
            "w": self.w,
            "u": {"u": self.u},
        }


def _all_positive_demands(inst: PLLRPInstance) -> List[float]:
    return [
        float(inst.demand[(k, t)])
        for k in inst.sr_ids
        for t in inst.periods
        if inst.demand[(k, t)] > 0
    ]


def compute_hat_R(inst: PLLRPInstance) -> Dict[int, float]:
    r"""Compute \hat{R}_i as described in the paper."""
    max_K = max(float(inst.module_capacity[m]) for m in inst.module_ids)
    n = int(ceil(inst.vehicle_capacity / max_K) + 1)
    hat_R: Dict[int, float] = {}
    for i in inst.pl_ids:
        others = [j for j in inst.pl_ids if j != i]
        others.sort(key=lambda j: inst.d_first(i, j))
        chosen = [inst.depot_id, i] + others[: max(0, n - 2)]
        def _xy(node: int) -> Tuple[float, float]:
            if node == inst.depot_id:
                return inst.coords[inst.depot_id]
            return inst.pl_coords[node]
        xs = [_xy(node)[0] for node in chosen]
        ys = [_xy(node)[1] for node in chosen]
        A_i = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if A_i <= 0:
            A_i = 1e-9
        L_hat = 0.98 * sqrt(n * A_i)
        hat_R[i] = L_hat / max(1, (n - 1))
    return hat_R


def solve_reduced_model_gurobi(inst: PLLRPInstance, time_limit_sec: int = 600) -> ReducedSolution:
    """Solve the reduced LFRS model with Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError(
            "gurobipy is required to solve the reduced model exactly. "
            "Install Gurobi in the target environment or use your existing solver stack."
        ) from exc

    hat_R = compute_hat_R(inst)
    positive_demands = _all_positive_demands(inst)
    _ = min(positive_demands) if positive_demands else 0.0
    K_max = max(float(inst.module_capacity[m]) for m in inst.module_ids)

    bigM_t = {
        t: min(
            inst.vehicle_capacity,
            K_max,
            sum(float(inst.demand[(k, tau)]) for k in inst.sr_ids for tau in inst.periods if tau >= t),
        )
        for t in inst.periods
    }

    access_id = str(WLS_PARAMS.get("WLSACCESSID", "")).strip()
    secret = str(WLS_PARAMS.get("WLSSECRET", "")).strip()
    try:
        license_id = int(WLS_PARAMS.get("LICENSEID", 0))
    except Exception:
        license_id = 0

    env = None
    if access_id and secret and license_id > 0:
        params = {
            "WLSACCESSID": access_id,
            "WLSSECRET": secret,
            "LICENSEID": license_id,
        }
        env = gp.Env(params=params)

    model = gp.Model("pl_lrp_lfrs_reduced", env=env) if env is not None else gp.Model("pl_lrp_lfrs_reduced")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit_sec

    z = model.addVars(inst.pl_ids, inst.module_ids, vtype=GRB.BINARY, name="z")
    y = model.addVars(inst.pl_ids, inst.sr_ids, vtype=GRB.BINARY, name="y")
    v = model.addVars(inst.pl_ids, inst.periods, vtype=GRB.BINARY, name="v")
    q = model.addVars(inst.pl_ids, inst.periods, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    I = model.addVars(inst.pl_ids, inst.periods, lb=0.0, vtype=GRB.CONTINUOUS, name="I")

    model.setObjective(
        gp.quicksum(inst.opening_cost[(i, m)] * z[i, m] for i in inst.pl_ids for m in inst.module_ids)
        + gp.quicksum(inst.alpha * hat_R[i] * v[i, t] for i in inst.pl_ids for t in inst.periods)
        + gp.quicksum(
            inst.beta * inst.d_pl_sr(i, k) * float(inst.demand[(k, t)]) * y[i, k]
            for i in inst.pl_ids
            for k in inst.sr_ids
            for t in inst.periods
        ),
        GRB.MINIMIZE,
    )

    for k in inst.sr_ids:
        model.addConstr(gp.quicksum(y[i, k] for i in inst.pl_ids) == 1, name=f"assign[{k}]")

    for i in inst.pl_ids:
        model.addConstr(gp.quicksum(z[i, m] for m in inst.module_ids) <= 1, name=f"one_module[{i}]")

    for i in inst.pl_ids:
        for idx, t in enumerate(inst.periods):
            prev_I = 0.0 if idx == 0 else I[i, inst.periods[idx - 1]]
            model.addConstr(
                prev_I + q[i, t] == gp.quicksum(float(inst.demand[(k, t)]) * y[i, k] for k in inst.sr_ids) + I[i, t],
                name=f"inv_bal[{i},{t}]",
            )

    for i in inst.pl_ids:
        for t in inst.periods:
            model.addConstr(
                gp.quicksum(float(inst.demand[(k, t)]) * y[i, k] for k in inst.sr_ids) + I[i, t]
                <= gp.quicksum(float(inst.module_capacity[m]) * z[i, m] for m in inst.module_ids),
                name=f"cap[{i},{t}]",
            )

    for i in inst.pl_ids:
        for t in inst.periods:
            model.addConstr(v[i, t] <= gp.quicksum(z[i, m] for m in inst.module_ids), name=f"repl_open[{i},{t}]")

    for i in inst.pl_ids:
        for t in inst.periods:
            model.addConstr(q[i, t] <= bigM_t[t] * v[i, t], name=f"q_if_v[{i},{t}]")

    for t in inst.periods:
        model.addConstr(
            gp.quicksum(float(inst.module_capacity[m]) * z[i, m] for i in inst.pl_ids for m in inst.module_ids)
            >= sum(float(inst.demand[(k, t)]) for k in inst.sr_ids) + gp.quicksum(I[i, t] for i in inst.pl_ids),
            name=f"agg_cap[{t}]",
        )

    model.optimize()

    status_map = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    status = status_map.get(model.Status, str(model.Status))
    if model.SolCount == 0:
        raise RuntimeError(f"Reduced model ended with status={status} and no feasible solution.")

    return ReducedSolution(
        z={(i, m): int(round(z[i, m].X)) for i in inst.pl_ids for m in inst.module_ids},
        y={(i, k): int(round(y[i, k].X)) for i in inst.pl_ids for k in inst.sr_ids},
        v={(i, t): int(round(v[i, t].X)) for i in inst.pl_ids for t in inst.periods},
        q={(i, t): float(q[i, t].X) for i in inst.pl_ids for t in inst.periods},
        I={(i, t): float(I[i, t].X) for i in inst.pl_ids for t in inst.periods},
        objective_value=float(model.ObjVal),
        status=status,
        runtime=float(model.Runtime),
        hat_R=hat_R,
    )


@dataclass
class _Route:
    nodes: List[int]
    load: float


class ClarkeWrightSolver:
    """Parallel Clarke-Wright savings for a capacitated single-depot VRP."""

    def __init__(self, depot_id: int = 0) -> None:
        self.depot_id = depot_id

    def solve(
        self,
        nodes: Sequence[int],
        demand: Mapping[int, float],
        dist: Mapping[Arc, float],
        capacity: float,
    ) -> List[List[int]]:
        nodes = [int(n) for n in nodes if demand.get(int(n), 0.0) > 1e-9]
        if not nodes:
            return []

        routes: Dict[int, _Route] = {n: _Route(nodes=[n], load=float(demand[n])) for n in nodes}
        node_to_route: Dict[int, int] = {n: n for n in nodes}

        savings: List[Tuple[float, int, int]] = []
        for idx, i in enumerate(nodes):
            for j in nodes[idx + 1 :]:
                s = dist[(self.depot_id, i)] + dist[(self.depot_id, j)] - dist[(i, j)]
                savings.append((s, i, j))
        savings.sort(reverse=True)

        for _, i, j in savings:
            ri_key = node_to_route.get(i)
            rj_key = node_to_route.get(j)
            if ri_key is None or rj_key is None or ri_key == rj_key:
                continue
            ri = routes[ri_key]
            rj = routes[rj_key]
            if ri.load + rj.load > capacity + 1e-9:
                continue

            new_nodes: Optional[List[int]] = None
            if ri.nodes[-1] == i and rj.nodes[0] == j:
                new_nodes = ri.nodes + rj.nodes
            elif ri.nodes[0] == i and rj.nodes[-1] == j:
                new_nodes = list(reversed(ri.nodes)) + list(reversed(rj.nodes))
            elif ri.nodes[0] == i and rj.nodes[0] == j:
                new_nodes = list(reversed(ri.nodes)) + rj.nodes
            elif ri.nodes[-1] == i and rj.nodes[-1] == j:
                new_nodes = ri.nodes + list(reversed(rj.nodes))

            if new_nodes is None:
                continue

            new_key = min(ri_key, rj_key)
            new_route = _Route(nodes=new_nodes, load=ri.load + rj.load)
            routes[new_key] = new_route
            del routes[max(ri_key, rj_key)]
            for node in new_nodes:
                node_to_route[node] = new_key

        return [[self.depot_id] + r.nodes + [self.depot_id] for r in routes.values()]


def _build_period_demands_from_reduced(reduced: ReducedSolution, inst: PLLRPInstance, t: int) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for i in inst.pl_ids:
        qty = float(reduced.q[(i, t)])
        if reduced.v[(i, t)] > 0 and qty > 1e-9:
            result[i] = qty
    return result


def _period_distances(inst: PLLRPInstance) -> Dict[Arc, float]:
    node_set = [inst.depot_id] + list(inst.pl_ids)
    return {(i, j): inst.d_first(i, j) for i in node_set for j in node_set if i != j}


def route_second_phase(inst: PLLRPInstance, reduced: ReducedSolution) -> WarmStartSolution:
    """Fix z,y,v,q,I and solve one VRP per period with Clarke-Wright."""
    dist = _period_distances(inst)
    cw = ClarkeWrightSolver(depot_id=inst.depot_id)

    warm = WarmStartSolution(
        z=dict(reduced.z),
        y=dict(reduced.y),
        v=dict(reduced.v),
        q=dict(reduced.q),
        I=dict(reduced.I),
        reduced_obj=reduced.objective_value,
        note="LFRS warm start generated from reduced model + Clarke-Wright period VRPs",
    )

    max_routes = 0
    for t in inst.periods:
        dem = _build_period_demands_from_reduced(reduced, inst, t)
        routes = cw.solve(list(dem.keys()), dem, dist, inst.vehicle_capacity)
        warm.period_routes[t] = routes
        max_routes = max(max_routes, len(routes))

        for route in routes:
            delivered = sum(dem[node] for node in route[1:-1])
            remaining = delivered
            prev = inst.depot_id
            for node in route[1:-1]:
                warm.x[(prev, node, t)] = 1
                warm.w[(node, t)] = remaining
                remaining -= dem[node]
                prev = node
            warm.x[(prev, inst.depot_id, t)] = 1

    for i in [inst.depot_id] + list(inst.pl_ids):
        for j in [inst.depot_id] + list(inst.pl_ids):
            if i == j:
                continue
            for t in inst.periods:
                warm.x.setdefault((i, j, t), 0)
    for i in inst.pl_ids:
        for t in inst.periods:
            warm.w.setdefault((i, t), 0.0)

    warm.u = max_routes
    return warm



def _extract_open_pls_from_warm(inst: PLLRPInstance, warm: WarmStartSolution) -> List[Dict[str, Any]]:
    opened: List[Dict[str, Any]] = []
    for i in inst.pl_ids:
        selected_module = None
        for m in inst.module_ids:
            if warm.z.get((i, m), 0) > 0:
                selected_module = m
                break
        if selected_module is not None:
            opened.append(
                {
                    "pl_id": int(i),
                    "module": int(selected_module),
                    "capacity": float(inst.module_capacity[selected_module]),
                    "fixed_cost": float(inst.opening_cost[(i, selected_module)]),
                }
            )
    return opened


def _route_arc_cost(inst: PLLRPInstance, warm: WarmStartSolution) -> float:
    total = 0.0
    for (i, j, t), value in warm.x.items():
        if value > 0:
            total += float(inst.alpha) * float(inst.d_first(i, j))
    return total


def _compensation_cost(inst: PLLRPInstance, warm: WarmStartSolution) -> float:
    total = 0.0
    for (i, k), value in warm.y.items():
        if value > 0:
            for t in inst.periods:
                total += float(inst.beta) * float(inst.d_pl_sr(i, k)) * float(inst.demand[(k, t)])
    return total


def summarize_lfrs_result(
    inst: PLLRPInstance,
    warm: WarmStartSolution,
    reduced: ReducedSolution,
    routing_runtime_seconds: float,
    time_limit_seconds: int,
) -> Dict[str, Any]:
    opened = _extract_open_pls_from_warm(inst, warm)
    fixed_pl_cost = sum(item["fixed_cost"] for item in opened)
    fleet_cost = float(inst.vehicle_fixed_cost) * int(warm.u)
    travel_cost = _route_arc_cost(inst, warm)
    compensation_cost = _compensation_cost(inst, warm)
    total_cost = fixed_pl_cost + fleet_cost + travel_cost + compensation_cost
    route_count_by_period = {str(t): len(warm.period_routes.get(t, [])) for t in inst.periods}
    total_routes = sum(route_count_by_period.values())
    return {
        "method": "LFRS",
        "status": str(reduced.status),
        "time_limit_seconds": int(time_limit_seconds),
        "reduced_runtime_seconds": float(reduced.runtime),
        "routing_runtime_seconds": float(routing_runtime_seconds),
        "total_runtime_seconds": float(reduced.runtime) + float(routing_runtime_seconds),
        "reduced_obj": float(reduced.objective_value),
        "final_total_cost": float(total_cost),
        "objective_breakdown": {
            "fixed_pl_cost": float(fixed_pl_cost),
            "fleet_cost": float(fleet_cost),
            "travel_cost": float(travel_cost),
            "compensation_cost": float(compensation_cost),
        },
        "fleet_size": int(warm.u),
        "route_count": int(total_routes),
        "route_count_by_period": route_count_by_period,
        "open_pl_count": len(opened),
        "opened_pls": opened,
    }


def build_lfrs_warm_start(inst: PLLRPInstance, reduced_time_limit_sec: int = 600) -> WarmStartSolution:
    reduced = solve_reduced_model_gurobi(inst, time_limit_sec=reduced_time_limit_sec)
    return route_second_phase(inst, reduced)


def run_lfrs(
    inst: PLLRPInstance,
    reduced_time_limit_sec: int = 600,
) -> Tuple[WarmStartSolution, ReducedSolution, Dict[str, Any]]:
    reduced = solve_reduced_model_gurobi(inst, time_limit_sec=reduced_time_limit_sec)
    routing_t0 = time.perf_counter()
    warm = route_second_phase(inst, reduced)
    routing_runtime_seconds = time.perf_counter() - routing_t0
    result_summary = summarize_lfrs_result(
        inst=inst,
        warm=warm,
        reduced=reduced,
        routing_runtime_seconds=routing_runtime_seconds,
        time_limit_seconds=reduced_time_limit_sec,
    )
    return warm, reduced, result_summary


def apply_starts_to_gurobi_model(
    model,
    warm: WarmStartSolution,
    z_vars: Mapping[ZKey, object],
    y_vars: Mapping[YKey, object],
    v_vars: Mapping[VTKey, object],
    q_vars: Mapping[QTKey, object],
    I_vars: Mapping[ITKey, object],
    x_vars: Mapping[XKey, object],
    w_vars: Mapping[WTKey, object],
    u_var: object,
) -> None:
    """Apply starts to an already-built gurobipy model."""
    for key, var in z_vars.items():
        var.Start = warm.z.get(key, 0)
    for key, var in y_vars.items():
        var.Start = warm.y.get(key, 0)
    for key, var in v_vars.items():
        var.Start = warm.v.get(key, 0)
    for key, var in q_vars.items():
        var.Start = warm.q.get(key, 0.0)
    for key, var in I_vars.items():
        var.Start = warm.I.get(key, 0.0)
    for key, var in x_vars.items():
        var.Start = warm.x.get(key, 0)
    for key, var in w_vars.items():
        var.Start = warm.w.get(key, 0.0)
    u_var.Start = warm.u
    model.update()


def _instance_metadata(
    inst: PLLRPInstance,
    instance_name: Optional[str] = None,
    source: str = "LFRS",
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    return {
        "instance_name": instance_name or "pl_lrp_instance",
        "source": source,
        "schema_version": schema_version,
        "periods": list(inst.periods),
        "module_ids": list(inst.module_ids),
        "pl_ids": list(inst.pl_ids),
        "sr_ids": list(inst.sr_ids),
        "vehicle_capacity": float(inst.vehicle_capacity),
        "vehicle_fixed_cost": float(inst.vehicle_fixed_cost),
    }


def _serialize_start_section(section: Mapping[Any, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for key, value in section.items():
        key_list = list(key) if isinstance(key, tuple) else [key]
        records.append({"key": key_list, "value": value})
    records.sort(key=lambda rec: tuple(rec["key"]))
    return records



def warm_start_to_json_dict(
    warm: WarmStartSolution,
    inst: PLLRPInstance,
    instance_name: Optional[str] = None,
    source: str = "LFRS",
    schema_version: str = "1.0",
    result_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "meta": _instance_metadata(inst=inst, instance_name=instance_name, source=source, schema_version=schema_version),
        "starts": {
            "z": _serialize_start_section(warm.z),
            "y": _serialize_start_section(warm.y),
            "v": _serialize_start_section(warm.v),
            "q": _serialize_start_section(warm.q),
            "I": _serialize_start_section(warm.I),
            "x": _serialize_start_section(warm.x),
            "w": _serialize_start_section(warm.w),
            "u": {"u": warm.u},
        },
        "diagnostics": {
            "reduced_obj": float(warm.reduced_obj),
            "note": warm.note,
            "period_routes": {str(t): routes for t, routes in warm.period_routes.items()},
        },
    }
    if result_summary is not None:
        payload["result_summary"] = dict(result_summary)
        payload["diagnostics"]["final_total_cost"] = float(result_summary.get("final_total_cost", warm.reduced_obj))
        if result_summary.get("total_runtime_seconds") is not None:
            payload["diagnostics"]["total_runtime_seconds"] = float(result_summary["total_runtime_seconds"])
        if result_summary.get("reduced_runtime_seconds") is not None:
            payload["diagnostics"]["reduced_runtime_seconds"] = float(result_summary["reduced_runtime_seconds"])
    return payload


def lfrs_result_to_json_dict(
    inst: PLLRPInstance,
    instance_name: Optional[str],
    result_summary: Mapping[str, Any],
    warm: WarmStartSolution,
    source: str = "LFRS",
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    return {
        "meta": _instance_metadata(inst=inst, instance_name=instance_name, source=source, schema_version=schema_version),
        "result_summary": dict(result_summary),
        "diagnostics": {
            "note": warm.note,
            "period_routes": {str(t): routes for t, routes in warm.period_routes.items()},
        },
    }


def validate_warm_start_against_instance(warm: WarmStartSolution, inst: PLLRPInstance) -> None:
    pl_set = set(inst.pl_ids)
    sr_set = set(inst.sr_ids)
    period_set = set(inst.periods)
    module_set = set(inst.module_ids)
    node_set = {inst.depot_id, *inst.pl_ids}

    for (i, m), _ in warm.z.items():
        if i not in pl_set:
            raise ValueError(f"Warm-start z references unknown PL {i}")
        if m not in module_set:
            raise ValueError(f"Warm-start z references unknown module {m}")

    for (i, k), _ in warm.y.items():
        if i not in pl_set:
            raise ValueError(f"Warm-start y references unknown PL {i}")
        if k not in sr_set:
            raise ValueError(f"Warm-start y references unknown SR {k}")

    merged_two_dim = {}
    merged_two_dim.update(warm.v)
    merged_two_dim.update(warm.q)
    merged_two_dim.update(warm.I)
    merged_two_dim.update(warm.w)
    for (i, t), _ in merged_two_dim.items():
        if i not in pl_set:
            raise ValueError(f"Warm-start references unknown PL {i}")
        if t not in period_set:
            raise ValueError(f"Warm-start references unknown period {t}")

    for (i, j, t), _ in warm.x.items():
        if i not in node_set or j not in node_set:
            raise ValueError(f"Warm-start x references unknown node(s) {(i, j)}")
        if i == j:
            raise ValueError(f"Warm-start x contains self-loop {(i, j, t)}")
        if t not in period_set:
            raise ValueError(f"Warm-start x references unknown period {t}")

    if warm.u < 0:
        raise ValueError("Warm-start fleet size u must be non-negative")


def save_warm_start_json(
    warm: WarmStartSolution,
    inst: PLLRPInstance,
    path: str | Path,
    instance_name: Optional[str] = None,
    source: str = "LFRS",
    schema_version: str = "1.0",
    result_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    validate_warm_start_against_instance(warm, inst)
    payload = warm_start_to_json_dict(
        warm=warm,
        inst=inst,
        instance_name=instance_name,
        source=source,
        schema_version=schema_version,
        result_summary=result_summary,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_instance_from_plain_dict(data: Mapping[str, Any]) -> PLLRPInstance:
    """Build PLLRPInstance from either the flat LFRS schema or the nested B&C schema."""

    def _pair_map(obj: Any) -> Dict[Tuple[int, int], float]:
        if isinstance(obj, Mapping):
            out: Dict[Tuple[int, int], float] = {}
            for k, v in obj.items():
                if isinstance(k, tuple) and len(k) == 2:
                    out[(int(k[0]), int(k[1]))] = float(v)
                else:
                    raise ValueError(f"Unsupported mapping key in pair map: {k!r}")
            return out
        out: Dict[Tuple[int, int], float] = {}
        for rec in obj:
            if len(rec) != 3:
                raise ValueError(f"Expected triple, got {rec}")
            out[(int(rec[0]), int(rec[1]))] = float(rec[2])
        return out

    def _coord_map(obj: Any) -> Dict[int, Tuple[float, float]]:
        if isinstance(obj, Mapping):
            return {int(k): (float(v[0]), float(v[1])) for k, v in obj.items()}
        out: Dict[int, Tuple[float, float]] = {}
        for rec in obj:
            if len(rec) != 3:
                raise ValueError(f"Expected triple, got {rec}")
            out[int(rec[0])] = (float(rec[1]), float(rec[2]))
        return out

    if "pl_ids" in data and "module_capacity" in data and "opening_cost" in data:
        return PLLRPInstance(
            pl_ids=[int(x) for x in data["pl_ids"]],
            sr_ids=[int(x) for x in data["sr_ids"]],
            periods=[int(x) for x in data["periods"]],
            module_ids=[int(x) for x in data["module_ids"]],
            module_capacity={int(k): float(v) for k, v in data["module_capacity"].items()},
            opening_cost=_pair_map(data["opening_cost"]),
            demand=_pair_map(data["demand"]),
            coords=_coord_map(data["coords"]),
            vehicle_capacity=float(data["vehicle_capacity"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            vehicle_fixed_cost=float(data.get("vehicle_fixed_cost", 0.0)),
            dist_first_echelon=data.get("dist_first_echelon"),
            dist_pl_sr=data.get("dist_pl_sr"),
            depot_id=int(data.get("depot_id", 0)),
        )

    required_bc_keys = {
        "periods",
        "modules",
        "depot",
        "candidate_pls",
        "service_regions",
        "vehicle_capacity",
        "travel_cost",
        "compensation_cost",
    }
    if required_bc_keys.issubset(set(data.keys())):
        periods = [int(t) for t in data["periods"]]
        module_capacity = {int(k): float(v) for k, v in data["modules"].items()}
        module_ids = sorted(module_capacity.keys())

        pl_ids: List[int] = []
        sr_ids: List[int] = []
        coords: Dict[int, Tuple[float, float]] = {0: (float(data["depot"]["x"]), float(data["depot"]["y"]))}
        pl_coords: Dict[int, Tuple[float, float]] = {}
        sr_coords: Dict[int, Tuple[float, float]] = {}
        opening_cost: Dict[Tuple[int, int], float] = {}
        demand: Dict[Tuple[int, int], float] = {}

        for item in data["candidate_pls"]:
            i = int(item["id"])
            pl_ids.append(i)
            pl_coords[i] = (float(item["x"]), float(item["y"]))
            coords[i] = pl_coords[i]
            fixed_costs = item["fixed_costs"]
            for m in module_ids:
                opening_cost[(i, m)] = float(fixed_costs[str(m)] if str(m) in fixed_costs else fixed_costs[m])

        for item in data["service_regions"]:
            k = int(item["id"])
            sr_ids.append(k)
            sr_coords[k] = (float(item["x"]), float(item["y"]))
            sr_demand = item["demand"]
            for t in periods:
                demand[(k, t)] = float(sr_demand[str(t)] if str(t) in sr_demand else sr_demand.get(t, 0.0))

        return PLLRPInstance(
            pl_ids=sorted(pl_ids),
            sr_ids=sorted(sr_ids),
            periods=periods,
            module_ids=module_ids,
            module_capacity=module_capacity,
            opening_cost=opening_cost,
            demand=demand,
            coords=coords,
            pl_coords=pl_coords,
            sr_coords=sr_coords,
            vehicle_capacity=float(data["vehicle_capacity"]),
            alpha=float(data["travel_cost"]),
            beta=float(data["compensation_cost"]),
            vehicle_fixed_cost=float(data.get("vehicle_fixed_cost", 0.0)),
            depot_id=0,
        )

    raise ValueError(
        "Unsupported input schema for LFRS. Expected either flat LFRS keys "
        "(pl_ids, module_capacity, opening_cost, ...) or nested B&C keys "
        "(modules, depot, candidate_pls, service_regions, ...)."
    )


__all__ = [
    "PLLRPInstance",
    "ReducedSolution",
    "WarmStartSolution",
    "compute_hat_R",
    "solve_reduced_model_gurobi",
    "route_second_phase",
    "build_lfrs_warm_start",
    "run_lfrs",
    "summarize_lfrs_result",
    "lfrs_result_to_json_dict",
    "apply_starts_to_gurobi_model",
    "build_instance_from_plain_dict",
    "warm_start_to_json_dict",
    "validate_warm_start_against_instance",
    "save_warm_start_json",
]
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LFRS warm start for PL-LRP instance"
    )
    parser.add_argument(
        "--instance",
        required=True,
        help="Path to input instance JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output warm-start JSON",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=int,
        default=1800,
        help="Reduced model time limit in seconds",
    )
    parser.add_argument(
        "--instance-name",
        default=None,
        help="Optional instance name",
    )
    args = parser.parse_args()

    input_path = Path(args.instance)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    inst = build_instance_from_plain_dict(data)

    warm, reduced, result_summary = run_lfrs(
        inst=inst,
        reduced_time_limit_sec=args.time_limit_seconds,
    )

    save_warm_start_json(
        warm=warm,
        inst=inst,
        path=args.output,
        instance_name=args.instance_name or input_path.stem,
        source="LFRS",
        schema_version="1.0",
        result_summary=result_summary,
    )

    print(json.dumps(result_summary, indent=2))


if __name__ == "__main__":
    main()