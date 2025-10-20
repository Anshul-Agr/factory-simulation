import os, json, time, copy, math, random, types
from typing import Dict, Any, List, Optional, Tuple, Iterable, Callable, Union
try:
    import yaml  
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False
import config as base_config  
from logging_export import export_all  
import main as sim_main
def _deepcopy_namespace(ns):
    class NS: pass
    out = NS()
    for k, v in ns.__dict__.items():
        if k.startswith("__") or isinstance(v, types.ModuleType):
            continue
        setattr(out, k, copy.deepcopy(v))
    return out
def _set_by_path(obj: Any, path: str, value: Any):
    parts = path.split('.')
    cur = obj
    for i, p in enumerate(parts[:-1]):
        if hasattr(cur, p):
            cur = getattr(cur, p)
        elif isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            bad_path = '.'.join(parts[:i+1])
            raise AttributeError(f"Error setting override: Cannot resolve path part '{p}' in '{bad_path}'. The container is not a valid object or dictionary.")
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    elif hasattr(cur, last):
        setattr(cur, last, value)
    else:
        raise AttributeError(f"Error setting override: The final container for path '{path}' is not a valid object or dictionary and cannot be assigned.")
def _merge_dicts(dst: Dict, src: Dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dicts(dst[k], v)
        else:
            dst[k] = v
def _apply_overrides(ns, overrides: Dict[str, Any]):
    if not overrides:
        return ns
    for key, override_value in overrides.items():
        if isinstance(key, str) and "." in key:
            try:
                _set_by_path(ns, key, override_value)
            except Exception as e:
                print(f"ERROR applying dot-key override '{key}': {e}")
                raise
            continue 
        setattr(ns, key, copy.deepcopy(override_value))
    return ns
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
def _load_json_or_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        if not _HAS_YAML:
            raise RuntimeError("YAML requested but PyYAML not installed")
        return yaml.safe_load(text) or {}
    return json.loads(text)
def _derive_seeds(policy: str, base_seed: Optional[int], n: int) -> List[Optional[int]]:
    if policy == "fixed":
        return [base_seed] * n
    if policy == "increment":
        start = 0 if base_seed is None else base_seed
        return [start + i for i in range(n)]
    if policy == "random":
        rng = random.Random(base_seed)
        return [rng.randrange(1, 10**9) for _ in range(n)]
    return [base_seed] * n
def _collect_key_metrics(outdir: str) -> Dict[str, Any]:
    k = {}
    kpis_path = os.path.join(outdir, "kpis.json")
    dashboard_path = os.path.join(outdir, "dashboard.json")
    try:
        with open(kpis_path, "r") as f:
            kpis = json.load(f)
        fin = kpis.get("financial", {})
        env = kpis.get("environment", {})
        k["total_revenue"] = fin.get("totalrevenue", 0.0)
        k["total_costs"] = fin.get("totalcosts", 0.0)
        k["net_profit"] = fin.get("totalrevenue", 0.0) - fin.get("totalcosts", 0.0)
        k["total_emissions_kg"] = env.get("totalemissionskg", 0.0)
        k["total_energy_kwh"] = env.get("totalenergykwh", 0.0)
    except Exception:
        pass
    try:
        with open(dashboard_path, "r") as f:
            dash = json.load(f)
        k["oeenowcast"] = dash.get("operational", {}).get("oeenowcast", 0.0)
        k["worker_util_latest"] = dash.get("workerutilization", 0.0)
    except Exception:
        pass
    return k
class ScenarioManager:
    def __init__(self, base_cfg_module=base_config):
        self.base_cfg_module = base_cfg_module
    def load(self, path: str) -> Dict[str, Any]:
        data = _load_json_or_yaml(path)
        return data or {}
    def apply_to_config(self, scenario: Dict[str, Any]):
        cfg_ns = _deepcopy_namespace(self.base_cfg_module)
        overrides = scenario.get("overrides", {})
        if not overrides and "config_overrides" in scenario:
            overrides = scenario["config_overrides"]
        _apply_overrides(cfg_ns, overrides)
        run = scenario.get("run", {})
        if "SIMULATION_TIME" in run:
            cfg_ns.SIMULATION_TIME = run["SIMULATION_TIME"]
        if "RANDOM_SEED" in run:
            cfg_ns.RANDOM_SEED = run["RANDOM_SEED"]
        return cfg_ns
    def run_once(self, scenario: Dict[str, Any], seed: Optional[int], output_dir: str, label: str) -> Dict[str, Any]:
        cfg_ns = self.apply_to_config(scenario)
        if seed is not None:
            cfg_ns.RANDOM_SEED = seed
        _ensure_dir(output_dir)
        print("SCENARIO PASS KEYS:", list(cfg_ns.PRODUCTS.keys())[:5])
        print("SCENARIO MATERIAL KEYS:", list(cfg_ns.MATERIAL_CONSUMPTION.keys()))
        summary = sim_main.run_simulation(config_ns=cfg_ns, output_dir=output_dir, scenario_label=label)
        metrics = _collect_key_metrics(output_dir)
        summary_out = {
            "label": label,
            "seed": seed,
            "output_dir": output_dir,
            "metrics": metrics,
            "summary": summary or {},
        }
        return summary_out
    def run_batch(self, scenarios: List[Dict[str, Any]], out_root: str) -> List[Dict[str, Any]]:
        results = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_root = os.path.join(out_root, f"batch_{timestamp}")
        _ensure_dir(batch_root)
        for i, sc in enumerate(scenarios):
            label = sc.get("name", f"scenario_{i+1}")
            run = sc.get("run", {})
            seed_policy = run.get("seed_policy", "increment")
            seeds = _derive_seeds(seed_policy, run.get("base_seed", None), 1)
            outdir = os.path.join(batch_root, label)
            res = self.run_once(sc, seeds[0], outdir, label)
            results.append(res)
        with open(os.path.join(batch_root, "index.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results
    def run_monte_carlo(self, scenario: Dict[str, Any], replications: int, out_root: str) -> Dict[str, Any]:
        run = scenario.get("run", {})
        seed_policy = run.get("seed_policy", "increment")
        seeds = _derive_seeds(seed_policy, run.get("base_seed", None), replications)
        label = scenario.get("name", "scenario")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mc_root = os.path.join(out_root, f"mc_{label}_{timestamp}")
        _ensure_dir(mc_root)
        per_run = []
        for i, sd in enumerate(seeds):
            outdir = os.path.join(mc_root, f"rep_{i+1:03d}")
            per_run.append(self.run_once(scenario, sd, outdir, f"{label}_rep{i+1:03d}"))
        agg = self._aggregate_mc(per_run)
        with open(os.path.join(mc_root, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)
        return {"root": mc_root, "replications": per_run, "aggregate": agg}
    def _aggregate_mc(self, per_run: List[Dict[str, Any]]) -> Dict[str, Any]:
        def stats(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
            sv = sorted(vals)
            n = len(vals)
            p90 = sv[min(n-1, int(0.9*(n-1)))]
            p50 = sv[min(n-1, int(0.5*(n-1)))]
            return {"mean": sum(vals)/n, "p50": p50, "p90": p90, "n": n}
        keys = ["oeenowcast", "net_profit", "total_costs", "total_revenue", "total_emissions_kg", "total_energy_kwh", "worker_util_latest"]
        cols: Dict[str, List[float]] = {k: [] for k in keys}
        for r in per_run:
            m = r.get("metrics", {})
            m = {**m, "net_profit": m.get("net_profit", 0.0)}  
            for k in keys:
                v = m.get(k, None)
                if isinstance(v, (int, float)):
                    cols[k].append(float(v))
        return {k: stats(vs) for k, vs in cols.items()}
    def sweep_parameters(self, scenario: Dict[str, Any], sweeps: Dict[str, List[Any]], out_root: str) -> List[Dict[str, Any]]:
        from itertools import product
        grid_keys = list(sweeps.keys())
        grid_vals = [sweeps[k] for k in grid_keys]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        root = os.path.join(out_root, f"sweep_{scenario.get('name','scenario')}_{timestamp}")
        _ensure_dir(root)
        outcomes = []
        idx = 0
        for combo in product(*grid_vals):
            idx += 1
            sc_copy = copy.deepcopy(scenario)
            overrides = sc_copy.get("overrides", {})
            for k, val in zip(grid_keys, combo):
                overrides[k] = val
            sc_copy["overrides"] = overrides
            label_bits = [f"{k.split('.')[-1]}={str(v)}" for k, v in zip(grid_keys, combo)]
            label = f"sweep_{idx:03d}_" + "_".join(label_bits)
            outdir = os.path.join(root, label)
            seed = sc_copy.get("run", {}).get("base_seed", None)
            res = self.run_once(sc_copy, seed, outdir, label)
            res["parameters"] = {k: v for k, v in zip(grid_keys, combo)}
            outcomes.append(res)
        rows = []
        for r in outcomes:
            base = {"label": r["label"], "seed": r["seed"], "output_dir": r["output_dir"]}
            base.update({f"param:{k}": v for k, v in r.get("parameters", {}).items()})
            for mk, mv in r.get("metrics", {}).items():
                base[f"metric:{mk}"] = mv
            rows.append(base)
        csv_path = os.path.join(root, "sweep_results.csv")
        from logging_export import writecsv  
        try:
            writecsv(csv_path, rows)
        except Exception:
            import csv
            with open(csv_path, "w", newline="") as f:
                if rows:
                    fieldnames = sorted(rows[0].keys())
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for row in rows:
                        w.writerow(row)
        with open(os.path.join(root, "sweep_index.json"), "w") as f:
            json.dump(outcomes, f, indent=2)
        return outcomes
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factory Simulation Scenario Runner")
    parser.add_argument(
        "--scenario",
        type=str,
        help="Path to a single scenario YAML or JSON file to run."
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to a scenario file containing a list of scenarios to run in a batch."
    )
    parser.add_argument(
        "--montecarlo",
        type=str,
        help="Path to a single scenario file to run for multiple replications."
    )
    parser.add_argument(
        "-n", "--replications",
        type=int,
        default=10,
        help="Number of replications for Monte Carlo run."
    )
    args = parser.parse_args()
    manager = ScenarioManager()
    if args.scenario:
        print(f"\\n--- RUNNING SINGLE SCENARIO: {args.scenario} ---")
        try:
            scenario_data = manager.load(args.scenario)
            output_root = scenario_data.get("run", {}).get("output_root", "data/processed")
            scenario_name = scenario_data.get("name", "unnamed_scenario")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_root, f"{scenario_name}_{timestamp}")
            run_config = scenario_data.get("run", {})
            seed = run_config.get("RANDOM_SEED", base_config.RANDOM_SEED)
            manager.run_once(
                scenario=scenario_data, 
                seed=seed,
                output_dir=output_dir,
                label=scenario_name
            )
            print(f"--- SCENARIO RUN COMPLETE. SEE OUTPUTS IN: {output_dir} ---")
        except Exception as e:
            print(f"\\nERROR: Failed to run scenario '{args.scenario}'.")
            print(f"DETAILS: {e}")
    elif args.batch:
        print(f"\\n--- RUNNING BATCH: {args.batch} ---")
        pass
    elif args.montecarlo:
        print(f"\\n--- RUNNING MONTE CARLO: {args.montecarlo} ({args.replications} reps) ---")
        pass
    else:
        print("\\nUsage: python scenarios.py --scenario <path_to_file.yaml>")
        print("No scenario specified. Exiting.")