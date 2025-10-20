import os
import csv
import json
import time
from typing import Dict, List, Any, Iterable, Optional
from collections import deque, defaultdict
import statistics
import math
import pandas as pd
class MetricsBus:
    def __init__(self, windows_hours=(4.0, 24.0)):
        self.gauges = {} 
        self.counters = defaultdict(float) 
        self.series = defaultdict(list) 
        self.windows = list(windows_hours)
        self.windowed = {w: defaultdict(deque) for w in self.windows} 
        self.alerts = [] 
        self._alert_index = {} 
    def set_gauge(self, name, value, t):
        self.gauges[name] = float(value)
        self._append(name, value, t)
    def inc(self, name, delta, t):
        self.counters[name] += float(delta)
        self._append(name, self.counters[name], t)
    def observe(self, name, value, t):
        self._append(name, value, t)
    def _append(self, name, value, t):
        t = float(t)
        v = float(value)
        self.series[name].append((t, v))
        for w in self.windows:
            dq = self.windowed[w][name]
            dq.append((t, v))
            t0 = t - w
            while dq and dq[0][0] < t0:
                dq.popleft()
    def window_stats(self, name, t, w):
        dq = self.windowed[w][name]
        if not dq:
            return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        vals = [v for _, v in dq]
        vals_sorted = sorted(vals)
        def perc(p):
            if not vals_sorted: return 0.0
            k = max(0, min(len(vals_sorted)-1, int(p/100 * (len(vals_sorted)-1))))
            return vals_sorted[k]
        return {
            "avg": statistics.fmean(vals) if vals else 0.0,
            "p50": perc(50),
            "p90": perc(90),
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
            "n": len(vals),
        }
    def slope(self, name, w):
        dq = self.windowed[w][name]
        if len(dq) < 2:
            return 0.0
        t0, v0 = dq[0]
        t1, v1 = dq[-1]
        dt = max(1e-9, t1 - t0)
        return (v1 - v0) / dt
    def raise_alert(self, key, severity, title, message, t, meta=None):
        if key in self._alert_index:
            idx = self._alert_index[key]
            self.alerts[idx]["latest_seen"] = t
            self.alerts[idx]["count"] += 1
            return
        rec = {
            "key": key, "severity": severity, "title": title, "message": message,
            "first_seen": t, "latest_seen": t, "count": 1, "meta": meta or {}
        }
        self._alert_index[key] = len(self.alerts)
        self.alerts.append(rec)
class JobLogger:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._eid: int = 0
    def clear(self) -> None:
        self.events.clear()
        self._eid = 0
    def __len__(self) -> int:
        return len(self.events)
    def log(self, event_type: str, sim_time: float, **fields: Any) -> None:
        self._eid += 1
        if 'stage_name' in fields:
            fields['stage'] = fields.pop('stage_name')
        row = {
            'event_id': self._eid,
            'event_type': event_type,
            'sim_time': sim_time,
            'timestamp': time.time(),
        }
        row.update(fields)
        self.events.append(row)
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
def _write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    rows = list(rows)
    if not rows and not fieldnames:
        open(path, 'w').close() 
        return
    if fieldnames is None:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        preferred_order = [
            'event_id', 'event_type', 'sim_time', 'timestamp', 'order_id',
            'item_id', 'job_id', 'product', 'stage', 'machine', 'type', 'duration'
        ]
        final_fieldnames = [k for k in preferred_order if k in all_keys]
        remaining_keys = sorted([k for k in all_keys if k not in preferred_order])
        final_fieldnames.extend(remaining_keys)
    else:
        final_fieldnames = fieldnames
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
        writer.writeheader()
        if rows:
            writer.writerows(rows)
def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
def export_transport_costs(financial_tracker, out_dir):
    if not hasattr(financial_tracker, "costevents"):
        return
    rows = [e for e in financial_tracker.costevents if e.get("type") == "transport"]
    if not rows:
        return
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "transport_costs.csv")
    fieldnames = ["simtime","type","mode","hours","km","cost"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
def _compute_operational_nowcast(machines, env_time):
    op = {"stages": {}, "plant": {}}
    oees = []
    try:
        from main import compute_oee_from_timeline
    except Exception:
        compute_oee_from_timeline = None
    for stage_name, m in machines.items():
        ms = m if isinstance(m, list) else [m]
        for mi in ms:
            tl = getattr(mi, "timeline", None)
            if tl and compute_oee_from_timeline:
                try:
                    k = compute_oee_from_timeline(tl, env_time)
                    oees.append(k.get("oee", 0.0))
                    op["stages"][getattr(mi, "name", f"{stage_name}")] = k
                except Exception:
                    pass
    plant_oee = sum(oees)/len(oees) if oees else 0.0
    op["plant"] = {"oee_nowcast": plant_oee}
    return op
def _compute_trends(metrics, env_time):
    if metrics is None or not hasattr(metrics, "windows") or not hasattr(metrics,"series") or not hasattr(metrics, "window_stats") or not hasattr(metrics, "slope"):
        return {"windows": {}}
    wins = {}
    for w in metrics.windows:
        wins[str(w)] = {
            "oee_slope": metrics.slope("oee", w),
            "wip_avg": metrics.window_stats("wip", env_time, w)["avg"],
            "worker_util_avg": metrics.window_stats("worker_utilization", env_time, w)["avg"],
            "queue_p90_by_stage": {}
        }
    for name in list(metrics.series.keys()):
        if name.startswith("queue:"):
            stage = name.split(":",1)[1]
            for w in metrics.windows:
                wins[str(w)]["queue_p90_by_stage"][stage] = metrics.window_stats(name, env_time, w)["p90"]
    return {"windows": wins}
def _evaluate_alerts(metrics, env_time, financial_summary, qualitytracker, environmental):
    if metrics is None or not hasattr(metrics, "series") or not hasattr(metrics, "raise_alert"):
        return []
    oee_now = metrics.series.get("oee", [])[-1][1] if metrics.series.get("oee") else None
    if oee_now is not None and oee_now < 0.55:
        metrics.raise_alert("oee_low", "high", "OEE below target", f"OEE {oee_now:.2f} < 0.55", env_time)
    for name, pts in metrics.series.items():
        if name.startswith("queue:"):
            for w in metrics.windows:
                dq = metrics.windowed[w][name]
                if dq and max(v for _, v in dq) >= 8:
                    stage = name.split(":",1)[1]
                    metrics.raise_alert(f"queue_high:{stage}", "medium", "Queue surge", f"{stage} queue >= 8", env_time, meta={"window_h": w})
    pm = metrics.gauges.get("profit_margin", 0.0)

    if (pm * 100) < 5.0:
        metrics.raise_alert("profit_margin_low", "high", "Profit margin low", f"Margin {pm*100:.1f}% < 5%", env_time)

    if qualitytracker is not None:
        try:
            k = qualitytracker.kpis()
            for prod, vals in k.get("by_product", {}).items():
                if vals.get("defect_rate", 0.0) > 0.08:
                    metrics.raise_alert(f"defect_spike:{prod}", "medium", "Defect rate high", f"{prod} defect rate {vals['defect_rate']:.2%}", env_time)
        except Exception:
            pass
    if environmental is not None:
        try:
            units = sum(environmental.product_units_completed.values()) or 1
            epu = environmental.total_emissions_kg / units
            if epu > 2.0:
                metrics.raise_alert("emissions_high", "low", "Emissions per unit high", f"{epu:.2f} kg/unit > 2.0", env_time)
        except Exception:
            pass
    return metrics.alerts
    
def export_all(output_dir: str,
               job_logger,
               machines,
               inventory_manager,
               order_tracker,
               financial_summary,
               financial_tracker,
               worker_pool=None,  
               environmental=None,
               metrics=None,
               job_metrics=None,
               qualitytracker=None,
               kpi_calculator=None,
               config_ns=None,
               advanced_financials=None,
               financial_viz_data=None) -> None:
    _ensure_dir(output_dir)
    warmup_period = getattr(config_ns, 'WARMUP_PERIOD', 0)
    print(f"Exporter engaged. Applying warm-up period of {warmup_period} hours.")
    job_events_df = pd.DataFrame(job_logger.events)
    post_warmup_jobs_df = job_events_df[job_events_df['sim_time'] >= warmup_period]
    completed_df = post_warmup_jobs_df[post_warmup_jobs_df['event_type'] == 'job_finished']
    avg_lead_time = completed_df['job_flow_time'].mean() if not completed_df.empty else 0.0
    print(f"Exporter engaged. Metrics object ID: {id(metrics)}")
    if job_logger:
        _write_csv(os.path.join(output_dir, 'job_events.csv'), job_logger.events)
    all_states = []
    if machines:
        for stage_name, machine_or_list in machines.items():
            machine_list = machine_or_list if isinstance(machine_or_list, list) else [machine_or_list]
            for machine in machine_list:
                if hasattr(machine, 'timeline') and hasattr(machine.timeline, 'get_as_dicts'):
                    states = machine.timeline.get_as_dicts()
                    for state_dict in states:
                        state_dict['stage'] = stage_name
                    all_states.extend(states)
    _write_csv(os.path.join(output_dir, 'machine_states.csv'), all_states)
    downtime_rows = []
    if machines:
        def dump_downtime(stage_name, mobj):
            for ev in getattr(mobj, 'downtime_events', []):
                row = {'stage': stage_name, 'machine': getattr(mobj, 'name', '')}
                row.update(ev)
                downtime_rows.append(row)
        for stage_name, m in machines.items():
            (dump_downtime(stage_name, mi) for mi in m) if isinstance(m, list) else dump_downtime(stage_name, m)
    _write_csv(os.path.join(output_dir, 'downtime.csv'), downtime_rows)
    if inventory_manager:
        _write_csv(os.path.join(output_dir, 'inventory_movements.csv'), getattr(inventory_manager, 'stock_movements', []))
        _write_csv(os.path.join(output_dir, 'procurement_orders.csv'), getattr(inventory_manager, 'procurement_orders', []))
        fg = [{'product': p, 'qty': q} for p, q in getattr(inventory_manager, 'finished_goods', {}).items()]
        _write_csv(os.path.join(output_dir, 'finished_goods.csv'), fg)
        shipped = [{'product': p, 'qty': q} for p, q in getattr(inventory_manager, 'shipped', {}).items()]
        _write_csv(os.path.join(output_dir, 'shipped.csv'), shipped)
    orders_rows = []
    if order_tracker:
        for oid, od in getattr(order_tracker, 'orders', {}).items():
            items = od.get('items', [])
            orders_rows.append({
                'order_id': oid, 'release_time': od.get('release_time'),
                'due_time': od.get('due_time'), 'start_time': od.get('start_time'),
                'completion_time': od.get('completion_time'), 'shipped_time': od.get('shipped_time'),
                'status': od.get('status'), 'items_completed': sum(1 for it in items if it.get('status') == 'completed'),
                'items_total': len(items)
            })
    _write_csv(os.path.join(output_dir, 'orders.csv'), orders_rows)
    if environmental:
        _write_csv(os.path.join(output_dir, "environment_states.csv"), getattr(environmental, "machine_state_energy", []))
        env_totals = [{"machine": m, "kwh": kwh, "co2_kg": environmental.machine_emissions_kg.get(m, 0.0)} for m, kwh in environmental.machine_energy_kwh.items()]
        _write_csv(os.path.join(output_dir, "environment_totals.csv"), env_totals)
        _write_csv(os.path.join(output_dir, "env_transport.csv"), getattr(environmental, "transport_events", []))
        _write_csv(os.path.join(output_dir, "env_materials.csv"), getattr(environmental, "material_events", []))
    if qualitytracker:
        _write_csv(os.path.join(output_dir, "quality_events.csv"), getattr(qualitytracker, 'quality_events', []))
        _write_csv(os.path.join(output_dir, "returns.csv"), getattr(qualitytracker, 'return_events', []))
        try:
            kpis = qualitytracker.kpis()
            summary = [{"product": p, **vals} for p, vals in kpis.get("by_product", {}).items()]
            _write_csv(os.path.join(output_dir, "quality_summary_by_product.csv"), summary)
            _write_json(os.path.join(output_dir, "quality_kpis.json"), kpis)
        except Exception:
            _write_json(os.path.join(output_dir, "quality_kpis.json"), {"by_product": {}}) 
    if advanced_financials:
        _write_json(os.path.join(output_dir, 'advanced_financials.json'), advanced_financials)
    if financial_viz_data:
        _write_json(os.path.join(output_dir, 'financial_viz_data.json'), financial_viz_data)
    env_time = financial_summary.get("simulation_time", financial_summary.get("simulationtime", 0.0)) or 0.0
    env_kpis = environmental.compute_kpis(env_time) if environmental else {}
    combined_kpis = {**financial_summary, "environment": env_kpis}
    _write_json(os.path.join(output_dir, "kpis.json"), combined_kpis)
    
    avg_worker_util = 0.0
    if metrics and 'worker_utilization' in metrics.series:
        util_series = metrics.series['worker_utilization']
        if util_series:
            
            total_weighted_value = 0
            last_ts = 0 
            last_value = util_series[0][1]

            
            for ts, value in util_series:
                duration = ts - last_ts
                if duration > 0:
                    total_weighted_value += last_value * duration
                last_ts = ts
                last_value = value
            
            
            final_duration = env_time - last_ts
            if final_duration > 0:
                total_weighted_value += last_value * final_duration
            
            if env_time > 0:
                avg_worker_util = total_weighted_value / env_time

    print(f"Dashboard: Average worker utilization calculated as {avg_worker_util:.3f}")

    latest_queues = {}
    if metrics:
        for k, pts in metrics.series.items():
            if k.startswith("queue:") and pts:
                latest_queues[k.split(":", 1)[1]] = pts[-1][1]
    dashboard = {
        "timestamp": time.time(),
        "sim_time": env_time,
        "operational": {
            "oee_nowcast": _compute_operational_nowcast(machines, env_time).get("plant", {}).get("oee_nowcast", 0.0),
            "queues": latest_queues,
            "worker_utilization": avg_worker_util
        },
        "quality": qualitytracker.kpis() if qualitytracker else {},
        "financial": financial_summary,
        "environment": env_kpis,
        "trends": _compute_trends(metrics, env_time),
        "alerts": _evaluate_alerts(metrics, env_time, financial_summary, qualitytracker, environmental),
        "analytical_comparison": kpi_calculator.analytical_comparison if kpi_calculator else {}
    }
    print("="*30 + " DEBUG 3: EXPORT INJECTION CHECK " + "="*30)
    if kpi_calculator:
        print(f"  'kpi_calculator' object was passed to export_all.")
        if hasattr(kpi_calculator, 'collect_analytical_results'):
            kpi_calculator.collect_analytical_results()
            print("  Called kpi_calculator.collect_analytical_results().")
        if hasattr(kpi_calculator, 'analytical_comparison') and kpi_calculator.analytical_comparison:
            print("  SUCCESS: 'analytical_comparison' dictionary is populated. Injecting into dashboard.json.")
            dashboard['analytical_comparison'] = kpi_calculator.analytical_comparison
        else:
            print("  [CRITICAL FAILURE] 'analytical_comparison' dictionary is EMPTY or does not exist.")
    else:
        print("  [CRITICAL FAILURE] 'kpi_calculator' object was NOT passed to export_all.")
    print("="*82 + "\n")
    _write_json(os.path.join(output_dir, "dashboard.json"), dashboard)
    print(f"SUCCESS: All simulation artifacts, including dashboard.json, exported to '{output_dir}'.")
def load_orders_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Orders CSV not found: {csv_path}")
    rows: List[Dict[str, Any]] = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            order_id = r.get('order_id')
            release_time = float(r.get('release_time', 0) or 0)
            product = r.get('product')
            qty = int(r.get('quantity', 1) or 1)
            due_in = r.get('due_in_hours')
            due_in_hours = float(due_in) if due_in not in (None, '',) else None
            rows.append({
                'order_id': order_id,
                'release_time': release_time,
                'product': product,
                'quantity': qty,
                'due_in_hours': due_in_hours
            })
    grouped: Dict[tuple, List[str]] = {}
    for r in rows:
        key = (r['order_id'], r['release_time'], r['due_in_hours'])
        grouped.setdefault(key, [])
        grouped[key].extend([r['product']] * r['quantity'])
    normalized: List[Dict[str, Any]] = []
    for (oid, rel, due), products in grouped.items():
        normalized.append({
            'order_id': oid,
            'release_time': rel,
            'products': products,
            'next_order_arrival_delay': 0.0,
            'due_in_hours': due
        })
    normalized.sort(key=lambda x: x['release_time'])
    return normalized