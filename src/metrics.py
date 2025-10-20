import statistics
from collections import defaultdict
import atexit
from config import ANALYTICAL_QUEUE_CONFIG
try:
    from main import FinancialTracker, QualityTracker, EnvironmentalTracker, PendingJobsTracker, StateTimeline
    from logging_export import MetricsBus
except ImportError:
    pass
class BottleneckDetector:
    def __init__(self, machines: dict, metrics_bus: 'MetricsBus', config: dict):
        self.machines = machines
        self.metrics_bus = metrics_bus
        self.config = config.get('bottleneck_rules', {})
        self.threshold_q_len = self.config.get('queue_length_threshold', 5)
        self.threshold_wait_time = self.config.get('wait_time_threshold_hours', 1.0)
        self.utilization_threshold = self.config.get('utilization_threshold_percent', 95.0)
    def analyze(self, current_time: float):
        self._analyze_queues(current_time)
        self._analyze_machine_utilization(current_time)
    def _analyze_queues(self, t: float):
        for stage_name, machine_list in self.machines.items():
            if not isinstance(machine_list, list):
                machine_list = [machine_list]
            total_queue_length = 0
            all_wait_times = []
            for machine in machine_list:
                if hasattr(machine, 'queue') and hasattr(machine.queue, 'jobs'):
                    queue_jobs = machine.queue.jobs
                    total_queue_length += len(queue_jobs)
                    for job in queue_jobs:
                        wait_time = t - job.arrival_time
                        all_wait_times.append(wait_time)
            if not all_wait_times:
                continue
            avg_wait_time = sum(all_wait_times) / len(all_wait_times)
            if total_queue_length > self.threshold_q_len and avg_wait_time > self.threshold_wait_time:
                self.metrics_bus.raise_alert(
                    key=f"bottleneck_queue_{stage_name}",
                    severity="high",
                    title=f"Bottleneck Suspected at {stage_name}",
                    message=f"Avg wait time is {avg_wait_time:.2f}h and current queue is {total_queue_length}.",
                    t=t,
                    meta={"stage": stage_name, "avg_wait": avg_wait_time, "current_q": total_queue_length}
                )
    def _analyze_machine_utilization(self, t: float):
        for stage_name, machine_list in self.machines.items():
            if not isinstance(machine_list, list):
                machine_list = [machine_list]
            for machine in machine_list:
                if not hasattr(machine, 'timeline'):
                    continue
                totals = machine.timeline.totals()
                running_time = totals.get('running', 0.0)
                total_time = t
                if total_time > 0:
                    utilization = (running_time / total_time) * 100
                    if utilization > self.utilization_threshold:
                        self.metrics_bus.raise_alert(
                            key=f"bottleneck_util_{machine.name}",
                            severity="medium",
                            title=f"High Utilization at {machine.name}",
                            message=f"Machine {machine.name} at {utilization:.1f}% utilization, may be a bottleneck.",
                            t=t,
                            meta={"machine": machine.name, "utilization": utilization}
                        )
class AlertsManager:
    def __init__(self, metrics_bus: 'MetricsBus', financial_tracker: 'FinancialTracker', qualitytracker: 'QualityTracker', config: dict):
        self.metrics_bus = metrics_bus
        self.financial_tracker = financial_tracker
        self.qualitytracker = qualitytracker
        self.config = config.get('alert_rules', {})
    def evaluate(self, current_time: float, oee_data: dict = None):
        self._check_financials(current_time)
        self._check_quality(current_time)
        if oee_data:
            self._check_operations(current_time, oee_data=oee_data)
    def _check_financials(self, t: float):
        profit_margin_threshold = self.config.get('profit_margin_threshold', 0.05)
        summary = self.financial_tracker.get_financial_summary(t)
        profit_margin = summary.get('profit_margin', 1.0)
        if profit_margin < profit_margin_threshold:
            self.metrics_bus.raise_alert(
                key="profit_margin_low",
                severity="high",
                title="Profit Margin Critically Low",
                message=f"Profit margin has fallen to {profit_margin:.2%}.",
                t=t,
                meta={'margin': profit_margin}
            )
    def _check_quality(self, t: float):
        defect_rate_threshold = self.config.get('defect_rate_threshold', 0.08)
        if not self.qualitytracker: return
        kpis = self.qualitytracker.kpis()
        for product, data in kpis.get('by_product', {}).items():
            defect_rate = data.get('defect_rate', 0.0)
            if defect_rate > defect_rate_threshold:
                self.metrics_bus.raise_alert(
                    key=f"defect_spike_{product}",
                    severity="medium",
                    title=f"Defect Rate Spike for {product}",
                    message=f"Defect rate for {product} is high at {defect_rate:.2%}.",
                    t=t,
                    meta={'product': product, 'defect_rate': defect_rate}
                )
    def _check_operations(self, t: float, oee_data: dict):
        oee_threshold = self.config.get('oee_threshold', 0.65)
        current_oee = oee_data.get('plant_oee_weighted_average', 1.0)
        if current_oee < oee_threshold:
            self.metrics_bus.raise_alert(
                key="oee_low",
                severity="high",
                title="Overall Plant OEE is Low",
                message=f"Plant OEE has dropped to {current_oee:.2f}.",
                t=t,
                meta={'oee': current_oee}
            )
class ComprehensiveKPIs:
    def __init__(self, metrics_bus, financial_tracker, quality_tracker, machines,worker_pool, **kwargs):
        self.metrics_bus = metrics_bus
        self.machines = machines
        self.worker_pool = worker_pool
        self.analytical_comparison = {}
        self.env = kwargs.get('env')
        self.job_metrics = kwargs.get('job_metrics')
        self.output_dir = kwargs.get('output_dir')
        self.cfg = kwargs.get('cfg')
        self.job_logger = kwargs.get('job_logger')
        self.trackers = {
            'financial': financial_tracker,
            'quality': quality_tracker
        }
    def calculate_and_export(self):
        print("\n--- Running single-station analysis ---")
        self.collect_analytical_results()
        print("--- Single-station analysis complete ---")
        print("\n--- Running Jackson Network analysis ---")
        if ANALYTICAL_QUEUE_CONFIG.get("jackson_network_enabled"):
            try:
                analyzer = JacksonNetworkAnalyzer(
                    all_stages=list(self.machines.keys()),
                    job_logger_events=self.job_logger.events,
                    machines=self.machines,
                    products_config=self.cfg.PRODUCTS
                )
                jackson_results = analyzer.solve(horizon=self.env.now)
                self.analytical_comparison['__jackson_network'] = jackson_results
                print("--- Jackson Network analysis successful ---")
            except Exception as e:
                print(f"!!! IGNORED ERROR during Jackson Network analysis: {e}")
                self.analytical_comparison['__jackson_network'] = {"error": f"Crashed: {e}"}
        print("\n--- All analytical calculations complete ---")
    def update_all(self, current_time: float):
        if current_time == 0: return
        oee_data = self._calculate_plant_oee(current_time)
        self.metrics_bus.set_gauge('plant_oee_nowcast', oee_data.get('plant_oee_weighted_average', 0), current_time)
        self.metrics_bus.set_gauge('plant_availability', oee_data.get('plant_availability', 0), current_time)
        self.metrics_bus.set_gauge('plant_performance', oee_data.get('plant_performance', 0), current_time)
        if self.worker_pool:
            total_workers = len(self.worker_pool.workers)
            if total_workers > 0:
                working_workers = sum(1 for w in self.worker_pool.workers.values() if w.is_working)
                utilization_rate = working_workers / total_workers
                self.metrics_bus.set_gauge('worker_utilization_rate', utilization_rate, current_time)
        if 'financial' in self.trackers:
            summary = self.trackers['financial'].get_financial_summary(current_time)
            self.metrics_bus.set_gauge('profit_per_hour', summary.get('profit_per_hour', 0), current_time)
            self.metrics_bus.set_gauge('net_profit', summary.get('net_profit', 0), current_time)
            self.metrics_bus.set_gauge('profit_margin', summary.get('profit_margin', 0), current_time)
    def _calculate_plant_oee(self, t: float) -> dict:
        from main import compute_oee_from_timeline 
        all_oee_data = []
        total_units = 0
        for stage_name, machine_list in self.machines.items():
            if not isinstance(machine_list, list): machine_list = [machine_list]
            for machine in machine_list:
                if hasattr(machine, 'timeline'):
                    units_produced = getattr(machine, 'units_produced_count', 0)
                    oee_stats = compute_oee_from_timeline(machine.timeline, horizon_end=t, good_units=units_produced, total_units=units_produced)
                    all_oee_data.append({'oee': oee_stats['oee'], 'availability': oee_stats['availability'], 'performance': oee_stats['performance'], 'units': units_produced})
                    total_units += units_produced
        if total_units == 0:
            return {'plant_oee_weighted_average': 0, 'plant_availability': 0, 'plant_performance': 0}
        weighted_oee = sum(d['oee'] * d['units'] for d in all_oee_data) / total_units
        weighted_avail = sum(d['availability'] * d['units'] for d in all_oee_data) / total_units
        weighted_perf = sum(d['performance'] * d['units'] for d in all_oee_data) / total_units
        return {
            'plant_oee_weighted_average': weighted_oee,
            'plant_availability': weighted_avail,
            'plant_performance': weighted_perf,
        }
    def collect_analytical_results(self):
        if not ANALYTICAL_QUEUE_CONFIG.get("enabled"):
            return
        simulated_wait_times = self._calculate_simulated_wait_times()
        for stage_name, machine_list in self.machines.items():
            if not isinstance(machine_list, list):
                machine_list = [machine_list]
            for m in machine_list:
                if hasattr(m, 'calculate_analytical_metrics'):
                    m.calculate_analytical_metrics()
                if hasattr(m, 'analytical_results') and m.analytical_results and not m.analytical_results.get("error"):
                    sim_wq = simulated_wait_times.get(m.stage, 0)
                    self.analytical_comparison[m.name] = {
                        "sim_ground_truth": {"avg_wait_time_Wq": sim_wq},
                        "analytical_prediction": m.analytical_results,
                        "parameters_used": m.analytical_params,
                    }
    def _calculate_simulated_wait_times(self):
        wait_times_by_stage = defaultdict(list)
        stage_starts = {}
        for event in self.job_logger.events:
            event_type = event.get('event_type')
            job_id = event.get('job_id')
            stage = event.get('stage')
            time = event.get('sim_time')
            if event_type == 'stage_queued' and all(k is not None for k in [job_id, stage, time]):
                stage_starts[(job_id, stage)] = time
            elif event_type == 'stage_start' and all(k is not None for k in [job_id, stage, time]):
                key = (job_id, stage)
                if key in stage_starts:
                    wait_time = time - stage_starts[key]
                    wait_times_by_stage[stage].append(wait_time)
        return {stage: statistics.mean(times) for stage, times in wait_times_by_stage.items() if times}
    def calculate_jackson_network(self):
        if not ANALYTICAL_QUEUE_CONFIG.get("jackson_network_enabled"):
            print("Jackson Network analysis is disabled in config.")
            return
        print("="*30 + "\nPERFORMING JACKSON NETWORK ANALYSIS\n" + "="*30)
        all_stages = sorted(list(self.machines.keys()))
        analyzer = JacksonNetworkAnalyzer(
            all_stages=all_stages,
            job_logger_events=self.job_logger.events,
            machines=self.machines,
            products_config=self.cfg.PRODUCTS
        )
        results = analyzer.solve(horizon=self.env.now)
        self.analytical_comparison['__jackson_network'] = results
        print("Jackson Network analysis complete.\n" + "="*30)
import numpy as np
from collections import defaultdict
import math
from queuing_system import mmc_model 
class JacksonNetworkAnalyzer:
    def __init__(self, all_stages: list, job_logger_events: list, machines: dict, products_config: dict):
        self.all_stages = sorted(list(set(all_stages)))
        self.events = job_logger_events
        self.machines = machines
        self.PRODUCTS = products_config
        self.stage_to_idx = {stage: i for i, stage in enumerate(self.all_stages)}
        self.idx_to_stage = {i: stage for i, stage in enumerate(self.all_stages)}
        self.num_stages = len(self.all_stages)
    def _calculate_routing_matrix(self):
        transition_counts = np.zeros((self.num_stages, self.num_stages))
        stage_departures = np.zeros(self.num_stages)
        jobs = defaultdict(list)
        for event in self.events:
            if event.get('event_type', '').startswith('stage_'):
                jobs[event.get('job_id')].append(event)
        for job_id, event_list in jobs.items():
            event_list.sort(key=lambda e: e['sim_time'])
            for i in range(len(event_list) - 1):
                if event_list[i].get('event_type') == 'stage_exit':
                    from_stage = event_list[i].get('stage')
                    from_idx = self.stage_to_idx.get(from_stage)
                    next_event = event_list[i+1]
                    if next_event.get('event_type') in ('stage_queued', 'stage_start'):
                        to_stage = next_event.get('stage')
                        to_idx = self.stage_to_idx.get(to_stage)
                        if from_idx is not None and to_idx is not None:
                            transition_counts[from_idx, to_idx] += 1
                            stage_departures[from_idx] += 1
        P = np.zeros((self.num_stages, self.num_stages))
        for i in range(self.num_stages):
            if stage_departures[i] > 0:
                P[i, :] = transition_counts[i, :] / stage_departures[i]
        return P
    def _calculate_external_arrivals(self, horizon):
        alpha = np.zeros(self.num_stages)
        first_stage_arrivals = defaultdict(int)
        product_to_first_stage = {
            p_name: p_config['routing'][0]
            for p_name, p_config in self.PRODUCTS.items() if 'routing' in p_config and p_config['routing']
        }
        for event in self.events:
            if event.get('event_type') == 'job_started':
                product_type = event.get('product')
                if product_type in product_to_first_stage:
                    stage = product_to_first_stage[product_type]
                    if stage in self.stage_to_idx:
                        first_stage_arrivals[stage] += 1
        for stage, count in first_stage_arrivals.items():
            idx = self.stage_to_idx[stage]
            alpha[idx] = count / horizon if horizon > 0 else 0
        return alpha
    def solve(self, horizon: float):
        if not self.events or horizon <= 0:
            return {"error": "Not enough simulation data or zero horizon."}
        P = self._calculate_routing_matrix()
        alpha = self._calculate_external_arrivals(horizon)
        if np.sum(alpha) == 0:
            return {"error": "No external arrivals were detected (Final Check)."}
        identity = np.identity(self.num_stages)
        try:
            effective_lambda = np.linalg.solve(identity - P.T, alpha)
        except np.linalg.LinAlgError:
            return {"error": "Could not solve traffic equations (singular matrix)."}
        node_results = {}
        total_system_wip = 0
        for i in range(self.num_stages):
            stage_name = self.idx_to_stage[i]
            lam = effective_lambda[i]
            machine_list = self.machines.get(stage_name)
            if not machine_list: continue
            if not isinstance(machine_list, list): machine_list = [machine_list]
            rep_machine = machine_list[0]
            service_times = rep_machine.service_times
            c = len(machine_list)
            mu = (1 / np.mean(service_times)) if service_times else 0
            node_metrics = mmc_model(lam, mu, c)
            cleaned_metrics = {}
            for key, value in node_metrics.items():
                if isinstance(value, (np.number)):
                    cleaned_metrics[key] = float(value)
                else:
                    cleaned_metrics[key] = value
            if 'error' not in cleaned_metrics:
                total_system_wip += cleaned_metrics.get('L', 0)
            node_results[stage_name] = {"effective_lambda": lam, "mu": mu, "c": c, **cleaned_metrics}
        return {
            "system_wip_analytical": total_system_wip,
            "nodes": node_results,
            "routing_matrix_P": P.tolist(),
            "external_arrivals_alpha": alpha.tolist()
        }