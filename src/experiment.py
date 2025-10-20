import os
import sys
import json
import time
import datetime
import argparse
import random
import statistics
import math
import numpy as np
import pandas as pd
import scenarios
import main as sim_main
import config as base_config
class ExperimentRunner:
    def __init__(self, scenario_manager, sim_run_func):
        self.sm = scenario_manager
        self.run_simulation = sim_run_func
    def run_experiments(self,
                        scenario_files: list[str],
                        replications: int,
                        output_root: str = "experiments"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(output_root, f"experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"--- Starting Experiment Run ---")
        print(f"Saving all outputs to: {experiment_dir}")
        all_scenario_results = {}
        for scenario_file in scenario_files:
            base_scen_cfg = self.sm.load(scenario_file)
            scenario_name = base_scen_cfg.get('name', os.path.basename(scenario_file).split('.')[0])
            print(f"\nProcessing Scenario: '{scenario_name}' for {replications} replications...")
            scenario_kpi_results = []
            master_seed = base_scen_cfg.get("run", {}).get("base_seed", int(time.time()))
            for i in range(replications):
                rep_num = i + 1
                replication_seed = master_seed + i
                run_cfg = self.sm.apply_to_config(base_scen_cfg)
                run_cfg.RANDOM_SEED = replication_seed
                replication_label = f"{scenario_name}_rep{rep_num:03d}"
                replication_output_dir = os.path.join(experiment_dir, replication_label)
                print(f"  Running replication {rep_num}/{replications} (Seed: {replication_seed})...")
                try:
                    kpi_summary = self.run_simulation(
                        config_ns=run_cfg,
                        output_dir=replication_output_dir,
                        scenario_label=replication_label
                    )
                    if kpi_summary:
                        scenario_kpi_results.append(kpi_summary)
                    else:
                        print(f"  WARNING: Replication {rep_num} for '{scenario_name}' returned no KPI summary.")
                except Exception as e:
                    print(f"  ERROR: Replication {rep_num} for '{scenario_name}' failed: {e}")
                    scenario_kpi_results.append({"error": str(e)})
            all_scenario_results[scenario_name] = scenario_kpi_results
        print("\n--- Experiment Complete. Analyzing results... ---")
        aggregated_stats = self._aggregate_and_analyze(all_scenario_results)
        summary_path = os.path.join(experiment_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(aggregated_stats, f, indent=4, cls=NumpyEncoder)
        print(f"\nFinal aggregated results saved to: {summary_path}")
        self._print_summary_table(aggregated_stats)
    def _aggregate_and_analyze(self, all_results: dict[str, list[dict]]) -> dict:
        from scipy import stats
        analysis_summary = {}
        for scenario_name, results_list in all_results.items():
            if not results_list:
                continue
            first_valid_result = next((r for r in results_list if r and 'error' not in r), None)
            if not first_valid_result:
                analysis_summary[scenario_name] = {"error": "No valid results found."}
                continue
            kpi_keys = first_valid_result.keys()
            aggregated_kpis = {key: [] for key in kpi_keys}
            for result in results_list:
                if result and 'error' not in result:
                    for key in kpi_keys:
                        aggregated_kpis[key].append(result.get(key, np.nan))
            scenario_stats = {}
            for key, values in aggregated_kpis.items():
                valid_values = [v for v in values if not np.isnan(v)]
                n = len(valid_values)
                if n > 1:
                    mean = statistics.mean(valid_values)
                    stdev = statistics.stdev(valid_values)
                    ci95_half_width = stats.t.ppf(0.975, df=n-1) * (stdev / math.sqrt(n))
                elif n == 1:
                    mean, stdev, ci95_half_width = valid_values[0], 0, 0
                else:
                    mean, stdev, ci95_half_width = 0, 0, 0
                scenario_stats[key] = {
                    "mean": mean,
                    "stdev": stdev,
                    "n": n,
                    "ci95_half_width": ci95_half_width,
                }
            analysis_summary[scenario_name] = scenario_stats
        return analysis_summary
    def _print_summary_table(self, aggregated_stats: dict):
        print("\n" + "="*80 + "\nSTATISTICAL EXPERIMENT SUMMARY\n" + "="*80)
        scenarios_list = list(aggregated_stats.keys())
        if not scenarios_list: return
        kpi_names = sorted(list(set(k for s in scenarios_list for k in aggregated_stats.get(s, {}).keys())))
        header = f"{'KPI':<25}" + "".join(f" | {name:<25}" for name in scenarios_list)
        print(header + "\n" + "-" * len(header))
        for kpi in kpi_names:
            row = f"{kpi:<25}"
            for name in scenarios_list:
                stats = aggregated_stats.get(name, {}).get(kpi)
                val_str = f"{stats['mean']:.3f} ± {stats['ci95_half_width']:.3f}" if stats and "mean" in stats else "N/A"
                row += f" | {val_str:<25}"
            print(row)
        print("="*80)
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)
def main():
    parser = argparse.ArgumentParser(description="Run simulation experiments with statistical rigor.")
    parser.add_argument('scenario_files', nargs='+', help="Paths to scenario config files.")
    parser.add_argument('-n', '--replications', type=int, default=10, help="Number of replications per scenario.")
    parser.add_argument('-o', '--output', type=str, default="experiments", help="Root directory for results.")
    args = parser.parse_args()
    if not all(os.path.exists(f) for f in args.scenario_files):
        print("FATAL: One or more scenario files not found.")
        sys.exit(1)
    try:
        scenario_manager = scenarios.ScenarioManager(base_config)
        runner = ExperimentRunner(scenario_manager, sim_main.run_simulation)
        runner.run_experiments(args.scenario_files, args.replications, args.output)
    except Exception as e:
        print(f"A critical error occurred during experiment execution: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()