import os
import json
import math
import csv
import io
import sys
import traceback
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
def resolve_latest_run(base_dir: str) -> str:
    try:
        if os.path.exists(os.path.join(base_dir, "dashboard.json")):
            return base_dir
        if not os.path.isdir(base_dir):
            return base_dir
        subdirs = []
        for name in os.listdir(base_dir):
            p = os.path.join(base_dir, name)
            if os.path.isdir(p):
                try:
                    mtime = os.path.getmtime(p)
                    subdirs.append((mtime, p))
                except Exception:
                    continue
        if not subdirs:
            return base_dir
        subdirs.sort(reverse=True)
        latest_path = subdirs[0][1]
        return latest_path
    except Exception:
        return base_dir
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False
REQUIRED_FILES = [
    "dashboard.json",          
    "kpis.json",               
    "job_events.csv",
    "machine_states.csv",
    "downtime.csv",
    "orders.csv",
    "inventory_movements.csv",
    "procurement_orders.csv",
    "finished_goods.csv",
    "shipped.csv",
    "environment_totals.csv",
    "env_transport.csv",
    "env_materials.csv",
    "quality_kpis.json"         
]
class FailLoudly(Exception):
    pass
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default
def perc(a: float, b: float) -> float:
    return (a / b * 100.0) if b and b != 0 else 0.0
def weighted_avg(values: List[float], weights: List[float]) -> float:
    s = sum(weights)
    return sum(v*w for v, w in zip(values, weights)) / s if s else 0.0
FILENAME_ALIASES = {
    "dashboard.json": ["dashboard.json"],
    "kpis.json": ["kpis.json"],
    "quality_kpis.json": ["quality_kpis.json", "qualitykpis.json"],
    "job_events.csv": ["job_events.csv", "jobevents.csv"],
    "machine_states.csv": ["machine_states.csv", "machinestates.csv"],
    "downtime_events.csv": ["downtime_events.csv", "downtime.csv"],
    "orders_summary.csv": ["orders_summary.csv", "orders.csv"],
    "inventory_movements.csv": ["inventory_movements.csv", "inventorymovements.csv"],
    "procurement_orders.csv": ["procurement_orders.csv", "procurementorders.csv"],
    "finished_goods.csv": ["finished_goods.csv", "finishedgoods.csv"],
    "shipped_goods.csv": ["shipped_goods.csv", "shipped.csv"],
    "environment_totals.csv": ["environment_totals.csv", "environmenttotals.csv"],
    "environment_states.csv": ["environment_states.csv", "environmentstates.csv"],
    "advanced_financials.json": ["advanced_financials.json"],
    "financial_viz_data.json": ["financial_viz_data.json"],
    "env_transport.csv": ["env_transport.csv", "envtransport.csv"],
    "env_materials.csv": ["env_materials.csv", "envmaterials.csv"],
    "quality_events.csv": ["quality_events.csv", "qualityevents.csv"],
    "returns.csv": ["returns.csv"],
    "transport_costs.csv": ["transport_costs.csv", "transportcosts.csv"]
}
REQUIRED_COLUMNS = {
    "job_events.csv": ["event_id", "event_type", "sim_time", "timestamp"],
    "machine_states.csv": ["start", "end", "state", "machine", "stage"],
    "environment_totals.csv": ["machine", "kwh", "co2_kg"],
    "orders_summary.csv": ["order_id", "items_completed", "items_total"]
}
HEADER_MAPS = {
    "job_events.csv": {
        "eventid": "event_id",
        "eventtype": "event_type",
        "simtime": "sim_time"
    },
    "machine_states.csv": {
        "Start": "start", "Begin": "start", "stime": "start",
        "End": "end", "Finish": "end", "etime": "end",
        "State": "state", "Status": "state", "label": "state",
        "Machine": "machine", "Resource": "machine",
        "Stage": "stage", "Station": "stage"
    },
    "environment_totals.csv": {
        "co2kg": "co2_kg"
    },
    "orders_summary.csv": {
        "itemscompleted": "items_completed",
        "itemstotal": "items_total"
    }
}
def _find_first_existing(output_dir: str, logical_name: str) -> Optional[str]:
    for candidate in FILENAME_ALIASES.get(logical_name, []):
        p = os.path.join(output_dir, candidate)
        if os.path.exists(p):
            return p
    return None
def read_json_any(output_dir: str, logical_name: str, required: bool) -> Any:
    p = _find_first_existing(output_dir, logical_name)
    if p is None:
        if required:
            raise FailLoudly(f"Missing required JSON: one of {FILENAME_ALIASES[logical_name]}")
        return None
    with open(p, "r") as f:
        return json.load(f)
def read_csv_any(output_dir: str, logical_name: str, required_cols: Optional[List[str]] = None) -> Any:
    p = _find_first_existing(output_dir, logical_name)
    if p is None:
        return None
    if os.path.getsize(p) == 0:
        if required_cols:
            return pd.DataFrame({c: pd.Series(dtype=float if c in ["start","end","sim_time","kwh","co2_kg","items_completed","items_total"] else "object")
                                 for c in required_cols})
        else:
            return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        if required_cols:
            df = pd.read_csv(p, header=None)
            if df.shape[1] >= len(required_cols):
                df = df.rename(columns={i: required_cols[i] for i in range(len(required_cols))})
            else:
                return pd.DataFrame({c: pd.Series(dtype=float if c in ["start","end","sim_time","kwh","co2_kg","items_completed","items_total"] else "object")
                                     for c in required_cols})
        else:
            return pd.DataFrame()
    norm = {c: (str(c).strip().lower().replace(" ", "_")) for c in df.columns}
    df.rename(columns=norm, inplace=True)
    m = HEADER_MAPS.get(logical_name, {})
    if m:
        df.rename(columns=m, inplace=True)
    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype=float if col in ["start","end","simtime","kwh","co2_kg","items_completed","items_total"] else "object")
    return df
def load_exports(output_dir: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["dashboard.json"] = read_json_any(output_dir, "dashboard.json", required=True)
    data["kpis.json"] = read_json_any(output_dir, "kpis.json", required=True)
    data["quality_kpis.json"] = read_json_any(output_dir, "quality_kpis.json", required=False)
    data['advanced_financials'] = read_json_any(output_dir, 'advanced_financials.json', required=False)
    data['financial_viz_data'] = read_json_any(output_dir, 'financial_viz_data.json', required=False)
    for logical, cols in [
        ("job_events.csv", REQUIRED_COLUMNS["job_events.csv"]),
        ("machine_states.csv", REQUIRED_COLUMNS["machine_states.csv"]),
        ("downtime_events.csv", None),
        ("orders_summary.csv", REQUIRED_COLUMNS["orders_summary.csv"]),
        ("inventory_movements.csv", None),
        ("procurement_orders.csv", None),
        ("finished_goods.csv", None),
        ("shipped_goods.csv", None),
        ("environment_totals.csv", REQUIRED_COLUMNS["environment_totals.csv"]),
        ("environment_states.csv", None),
        ("env_transport.csv", None),
        ("env_materials.csv", None),
        ("quality_events.csv", None),
        ("returns.csv", None),
        ("transport_costs.csv", None),
    ]:
        data[logical] = read_csv_any(output_dir, logical, cols)
    return data
def _g(obj, *paths, default=None):
    for p in paths:
        cur = obj
        ok = True
        for k in p.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return default
def compute_core_metrics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    dash = bundle.get("dashboard.json") or {}
    kpis = bundle.get("kpis.json") or {}
    simtime = safe_float(_g(dash, "sim_time_hours", "simtime_hours", "sim_time", default=0.0))
    print("[dbg] sim_time_hours =", simtime)
    oee_nowcast = safe_float(_g(dash, "oee_nowcast", "operational.oee_nowcast", default=0.0))
    worker_utilization = safe_float(_g(
        dash,
        "worker_utilization",
        "operational.worker_utilization",
        "labor.utilization",
        "labor.utilization_rate",
        "labor.utilization_percent",
        default=0.0
    ))
    if worker_utilization > 1.0:
        worker_utilization = worker_utilization / 100.0
    print("[dbg] worker_utilization raw%", worker_utilization)
    fin = _g(dash, "financial") or _g(kpis, "financial") or {}
    total_revenue = safe_float(_g(fin, "total_revenue", "revenue_total", "totalrevenue", default=0.0))
    total_costs   = safe_float(_g(fin, "total_costs", "costs_total", "totalcosts",   default=0.0))
    net_profit    = safe_float(_g(fin, "net_profit", "netprofit", default=total_revenue - total_costs))
    rev_hr_raw = _g(fin, "revenue_per_hour", default=None)
    prof_hr_raw = _g(fin, "profit_per_hour", default=None)
    def coerce_pos(x):
        try:
            v = float(x)
            return v if v > 0 else None
        except Exception:
            return None
    revenue_per_hour = coerce_pos(rev_hr_raw)
    profit_per_hour  = coerce_pos(prof_hr_raw)
    if revenue_per_hour is None:
        revenue_per_hour = (total_revenue / simtime) if simtime else 0.0
    if profit_per_hour is None:
        profit_per_hour  = (net_profit / simtime) if simtime else 0.0
    total_downtime_cost = safe_float(_g(fin, "total_downtime_cost", "downtime_cost_total", default=0.0))
    profit_margin = perc(net_profit, total_revenue)
    print("[dbg] core rates computed:", revenue_per_hour, profit_per_hour)
    env = _g(dash, "environment") or _g(kpis, "environment") or {}
    total_emissions_kg = safe_float(_g(env, "total_emissions_kg", "emissions_kg_total", default=0.0))
    total_energy_kwh = safe_float(_g(env, "total_energy_kwh", "energy_kwh_total", default=0.0))
    quality = _g(dash, "quality") or {}
    by_product = _g(quality, "by_product", "get_by_product", default={})
    trends = _g(dash, "trends", default={})
    alerts = _g(dash, "alerts", default=[])
    return {
        "sim_time_hours": simtime,
        "oee_nowcast": oee_nowcast,
        "worker_utilization": worker_utilization,
        "total_revenue": total_revenue,
        "total_costs": total_costs,
        "total_downtime_cost": total_downtime_cost,
        "net_profit": net_profit,
        "profit_margin": profit_margin,
        "revenue_per_hour": revenue_per_hour,
        "profit_per_hour": profit_per_hour,
        "total_emissions_kg": total_emissions_kg,
        "total_energy_kwh": total_energy_kwh,
        "quality_by_product": by_product,
        "quality_summary_present": bool(quality),
        "trends": trends,
        "alerts": alerts,
    }
STATE_MAP = {
    "waiting_worker": "waitingworker",
    "waiting material": "waitingmaterial",
    "calendar_off": "calendaroff",
    "CalendarOff": "calendaroff",
    "Running": "running",
    "Setup": "setup",
    "Idle": "idle",
    "Maintenance": "maintenance",
    "Breakdown": "breakdown",
    "waitingworker": "waitingworker",
    "waiting_worker": "waitingworker",
    "waiting worker": "waitingworker",
    "waitingmaterial": "waitingmaterial",
    "waiting_material": "waitingmaterial",
    "waiting material": "waitingmaterial",
    "calendar_off": "calendaroff",
    "calendaroff": "calendaroff",
    "offcalendar": "calendaroff",
    "run": "running",
    "prod": "running",
    "setup_changeover": "setup",
    "changeover": "setup",
    "maint": "maintenance",
    "down": "breakdown",
    "break": "idle"
}
def compute_oee_from_machinestates(df_states: pd.DataFrame, horizon_end: Optional[float]) -> Dict[str, Any]:
    if df_states is None or isinstance(df_states, str) or df_states.empty:
        return {"oee": 0.0, "availability": 0.0, "performance": 0.0}
    df = df_states.copy()
    for c in ["state","machine","stage"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().str.replace(" ", "_")
            df[c] = df[c].map(lambda x: STATE_MAP.get(x, x))
    for col in ["start", "end"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start","end"])
    df = df[df["end"] > df["start"]]
    if horizon_end is not None and horizon_end > 0:
        df["end"] = df["end"].clip(upper=float(horizon_end))
        df = df[df["end"] > df["start"]]
    df["duration"] = (df["end"] - df["start"]).astype(float)
    totals = df.groupby("state", dropna=False)["duration"].sum().to_dict()
    running = totals.get("running", 0.0)
    setup = totals.get("setup", 0.0)
    idle = totals.get("idle", 0.0)
    maintenance = totals.get("maintenance", 0.0)
    breakdown = totals.get("breakdown", 0.0)
    calendaroff = totals.get("calendaroff", 0.0)
    total_time = running + setup + idle + totals.get("waitingworker", 0.0) + totals.get("waitingmaterial", 0.0) + maintenance + breakdown + calendaroff
    planned_time = max(total_time - calendaroff, 0.0)
    uptime = max(planned_time - (maintenance + breakdown), 0.0)
    availability = perc(uptime, planned_time)
    performance = perc(running, max(running + setup + idle + totals.get("waitingworker", 0.0) + totals.get("waitingmaterial", 0.0), 0.0))
    quality = 100.0
    oee = (availability/100.0) * (performance/100.0) * (quality/100.0) * 100.0
    return {
        "oee": oee,
        "availability": availability,
        "performance": performance,
        "quality_assumed_percent": quality,
        "planned_time_hours": planned_time,
        "uptime_hours": uptime,
        "waiting_worker_hours": totals.get("waitingworker", 0.0),
        "waiting_material_hours": totals.get("waitingmaterial", 0.0)
    }
def compute_financial_deep(fin_summary: Dict[str, Any], df_downtime: Any, df_orders: Any) -> Dict[str, Any]:
    total_revenue = safe_float(fin_summary.get("total_revenue", 0.0))
    total_costs = safe_float(fin_summary.get("total_costs", 0.0))
    net_profit = total_revenue - total_costs
    profit_margin = perc(net_profit, total_revenue)
    fill_rate_percent = None
    if isinstance(df_orders, pd.DataFrame) and not df_orders.empty:
        ic = "items_completed" if "items_completed" in df_orders.columns else None
        it = "items_total"     if "items_total" in df_orders.columns     else None
        if ic and it:
            done = pd.to_numeric(df_orders[ic], errors="coerce").fillna(0).sum()
            total = pd.to_numeric(df_orders[it], errors="coerce").fillna(0).sum()
            fill_rate_percent = perc(done, total)
    downtime_cost_recorded = safe_float(fin_summary.get("total_downtime_cost", 0.0))
    if isinstance(df_downtime, pd.DataFrame) and not df_downtime.empty:
        for candidate in ["total_cost", "totalcost", "cost_total", "cost"]:
            if candidate in df_downtime.columns:
                dt_cost_sum = pd.to_numeric(df_downtime[candidate], errors="coerce").fillna(0.0).sum()
                downtime_cost_recorded = max(downtime_cost_recorded, dt_cost_sum)
                break
    return {
        "total_revenue": total_revenue,
        "total_costs": total_costs,
        "net_profit": net_profit,
        "profit_margin": profit_margin,
        "downtime_cost_total": downtime_cost_recorded,
        "order_fill_rate_percent": fill_rate_percent
    }
def compute_environment_deep(env_json: Dict[str, Any], df_env_totals: Any) -> Dict[str, Any]:
    total_emissions_kg = safe_float(env_json.get("total_emissions_kg", 0.0))
    total_energy_kwh = safe_float(env_json.get("total_energy_kwh", 0.0))
    if isinstance(df_env_totals, pd.DataFrame) and not df_env_totals.empty:
        e_kwh = pd.to_numeric(df_env_totals.get("kwh", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        e_co2 = pd.to_numeric(df_env_totals.get("co2kg", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        total_energy_kwh = max(total_energy_kwh, e_kwh)
        total_emissions_kg = max(total_emissions_kg, e_co2)
    return {
        "total_emissions_kg": total_emissions_kg,
        "total_energy_kwh": total_energy_kwh
    }
def compute_worker_util_from_jobs(df_jobs: Any, horizon_end: Optional[float]) -> Optional[float]:
    if not isinstance(df_jobs, pd.DataFrame) or df_jobs.empty:
        return None
    df = df_jobs.copy()
    df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns}, inplace=True)
    has_start_end = "start" in df.columns and "end" in df.columns
    if has_start_end:
        s = pd.to_numeric(df["start"], errors="coerce")
        e = pd.to_numeric(df["end"], errors="coerce")
        if horizon_end:
            e = e.clip(upper=float(horizon_end))
        dur = (e - s).clip(lower=0).fillna(0.0)
    else:
        if "event_type" in df.columns and "duration" in df.columns:
            dur = pd.to_numeric(df.loc[df["event_type"] == "busy", "duration"], errors="coerce").fillna(0.0)
        else:
            return None
    busy = float(dur.sum())
    planned = float(horizon_end) if horizon_end and horizon_end > 0 else None
    if not planned or planned <= 0:
        return None
    util = max(0.0, min(1.0, busy / planned))
    return util
def assemble_dashboard(output_dir: str) -> Dict[str, Any]:
    print(f"--- ABSOLUTE PROOF: ASSEMBLE_DASHBOARD IS USING THIS DIRECTORY ---")
    print(output_dir)
    print("--------------------------------------------------------------------")

    bundle = load_exports(output_dir)
    core = compute_core_metrics(bundle)
    df_states = bundle.get("machine_states.csv")
    oee_validation = compute_oee_from_machinestates(df_states, core["sim_time_hours"]) if isinstance(df_states, pd.DataFrame) else {
        "oee": None, "availability": None, "performance": None
    }
    fin_json = {
        "total_revenue": core["total_revenue"],
        "total_costs": core["total_costs"],
        "net_profit": core["net_profit"],
        "total_downtime_cost": core.get("total_downtime_cost", _g(bundle.get("dashboard.json") or {}, "financial.total_downtime_cost", "financial.downtime_cost_total", default=0.0)),
    }
    def pick_df(bundle, a, b):
        x = bundle.get(a, None)
        if isinstance(x, pd.DataFrame):
            return x
        y = bundle.get(b, None)
        return y if isinstance(y, pd.DataFrame) else None
    df_downtime = pick_df(bundle, "downtime_events.csv", "downtime.csv")
    df_orders   = pick_df(bundle, "orders_summary.csv", "orders.csv")
    fin_deep = compute_financial_deep(fin_json, df_downtime, df_orders)
    env_json = bundle.get("dashboard.json", {}).get("environment") or bundle.get("kpis.json", {}).get("environment") or {}
    df_env_totals = bundle.get("environment_totals.csv")
    env_deep = compute_environment_deep(env_json, df_env_totals)
    quality = bundle.get("dashboard.json", {}).get("quality") or bundle.get("quality_kpis.json") or {}
    inv = {}
    df_inv = bundle.get("inventory_movements.csv")
    if isinstance(df_inv, pd.DataFrame) and not df_inv.empty:
        cols = [c for c in ["material", "delta", "quantity", "type"] if c in df_inv.columns]
        inv["has_inventory"] = True
        inv["columns_present"] = cols
        if "material" in df_inv.columns:
            qty_col = "delta" if "delta" in df_inv.columns else ("quantity" if "quantity" in df_inv.columns else None)
            if qty_col:
                s = pd.to_numeric(df_inv[qty_col], errors="coerce").fillna(0.0)
                inv["net_movement_by_material"] = df_inv.groupby("material")[qty_col].apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0.0).sum()).to_dict()
    else:
        inv["has_inventory"] = False
    print("[dbg] fin_deep:", fin_deep)
    print("[dbg] core rates:", core["revenue_per_hour"], core["profit_per_hour"])
    print("[dbg] types:", type(df_downtime).__name__, type(df_orders).__name__)
    worker_util_eff = core["worker_utilization"]
    if worker_util_eff <= 0.0:
        ww = oee_validation.get("waiting_worker_hours") if isinstance(oee_validation, dict) else None
        pt = oee_validation.get("planned_time_hours") if isinstance(oee_validation, dict) else None
        if ww is not None and pt and pt > 0:
            worker_util_eff = max(0.0, min(1.0, 1.0 - float(ww)/float(pt)))
        if worker_util_eff <= 0.0:
            df_jobs = bundle.get("job_events.csv")
            util_jobs = compute_worker_util_from_jobs(df_jobs, core["sim_time_hours"])
            if util_jobs is not None:
                worker_util_eff = util_jobs
    out = {
        "sim_time_hours": core["sim_time_hours"],
        "oee_nowcast": core["oee_nowcast"],
        "oee_validation": oee_validation,
        "worker_utilization": worker_util_eff,
        "financial": {
            "total_revenue": fin_deep["total_revenue"],
            "total_costs": fin_deep["total_costs"],
            "net_profit": fin_deep["net_profit"],
            "profit_margin": fin_deep["profit_margin"],
            "downtime_cost_total": fin_deep["downtime_cost_total"],
            "order_fill_rate_percent": fin_deep["order_fill_rate_percent"],
            "revenue_per_hour": core["revenue_per_hour"],
            "profit_per_hour": core["profit_per_hour"]
        },
        "environment": {
            "total_emissions_kg": env_deep["total_emissions_kg"],
            "total_energy_kwh": env_deep["total_energy_kwh"]
        },
        "quality": quality,
        "trends": core["trends"],
        "alerts": core["alerts"],
        "inventory": inv,
        "_source_dir": output_dir
    }
    out['analytical_comparison'] = bundle.get("dashboard.json", {}).get("analytical_comparison")
    out['advanced_financials'] = bundle.get('advanced_financials')
    out['financial_viz_data'] = bundle.get('financial_viz_data')
    print("[dbg] sim_time_hours =", core["sim_time_hours"])
    print("[dbg] totals:", core["total_revenue"], core["total_costs"], core["net_profit"])
    print("[dbg] states rows =", 0 if not isinstance(df_states, pd.DataFrame) else len(df_states))
    print("[dbg] fin_deep_fixed:", fin_deep)
    print("[dbg] core rates:", core["revenue_per_hour"], core["profit_per_hour"])
    return out
app = FastAPI(title="Factory Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
@app.get("/api/dashboard")
def get_dashboard(output_dir: str):
    try:
        resolved = resolve_latest_run(output_dir)
        result = assemble_dashboard(resolved)
        return result
    except FailLoudly as e:
        raise HTTPException(status_code=500, detail=f"Loud failure: {e}")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}\n{tb}")
def run_streamlit(output_dir: str):
    st.set_page_config(layout="wide")
    st.title("Industrial Factory Dashboard")
    try:
        resolved = resolve_latest_run(output_dir)
        assembled = assemble_dashboard(resolved)
        st.write({"dbg_financial": assembled.get("financial")})
    except Exception as e:
        st.error(f"Failed to assemble dashboard: {e}")
        st.stop()
    fin = assembled.get("financial", {}) or {}
    env = assembled.get("environment", {}) or {}
    oee_nowcast = float(assembled.get("oee_nowcast") or 0.0)
    worker_util = float(assembled.get("worker_utilization") or 0.0)
    net_profit = float(fin.get("net_profit") or 0.0)
    profit_margin = float(fin.get("profit_margin") or 0.0)
    rev_hr = float(fin.get("revenue_per_hour") or assembled.get("revenue_per_hour") or 0.0)
    prof_hr = float(fin.get("profit_per_hour") or assembled.get("profit_per_hour") or 0.0)
    emissions = float(env.get("total_emissions_kg") or 0.0)
    energy = float(env.get("total_energy_kwh") or 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OEE Nowcast %", f"{oee_nowcast:.1f}")
    c2.metric("Worker Util %", f"{worker_util*100:.1f}")
    c3.metric("Net Profit", f"{net_profit:.2f}")
    c4.metric("Profit Margin %", f"{profit_margin:.1f}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Revenue/hr", f"{rev_hr:.2f}")
    c6.metric("Profit/hr", f"{prof_hr:.2f}")
    c6.metric("Downtime Cost", f"{float(fin.get('downtime_cost_total') or 0.0):.2f}")
    c7.metric("Emissions kg", f"{emissions:.2f}")
    c8.metric("Energy kWh", f"{energy:.2f}")
    st.subheader("OEE Validation (from machine states)")
    oeev = assembled["oee_validation"]
    st.write({
        "oee_percent": oeev.get("oee"),
        "availability_percent": oeev.get("availability"),
        "performance_percent": oeev.get("performance"),
        "planned_time_hours": oeev.get("planned_time_hours"),
        "uptime_hours": oeev.get("uptime_hours")
    })
    st.subheader("Alerts")
    alerts = assembled.get("alerts") or []
    if alerts:
        st.dataframe(pd.DataFrame(alerts))
    else:
        st.success("No alerts.")
    st.subheader("Quality KPIs")
    q = assembled.get("quality") or {}
    if q:
        st.json(q)
    else:
        st.info("Quality KPIs not available.")
    st.subheader("Inventory")
    inv = assembled.get("inventory") or {}
    st.json(inv)
    st.subheader("🔬 Queueing Theory vs. Simulation Analysis")
    analytical_data = assembled.get("analytical_comparison")
    
    if analytical_data:
        st.info(
            """
            This section compares performance metrics predicted by mathematical **analytical models**
            against the **simulation ground truth** for each individual machine. Large differences 
            can reveal where the simulation's complexities (like breakdowns or worker logic) 
            deviate from the ideal model's assumptions.
            """
        )
    
        machine_keys = [k for k in analytical_data.keys() if k != "__jackson_network"]
    
        if not machine_keys:
            st.warning(
                "**No Data for Individual Machine Analysis**\n\n"
                "The simulation did not generate enough data to automatically parameterize the analytical models "
                "for any individual machine. This is a common and normal outcome in scenarios with low machine utilization "
                "or when a large number of parallel machines share the workload."
            )
        else:
            for machine_name in sorted(machine_keys):
                data = analytical_data[machine_name]
                model_type = data.get('analytical_prediction', {}).get('model', 'N/A')
    
                with st.expander(f"**Machine: {machine_name}** (Model: **{model_type}**)", expanded=True):
                    
                    sim_data = data.get('sim_ground_truth', {})
                    prediction_data = data.get('analytical_prediction', {})
                    params = data.get('parameters_used', {})
                    
                   
                    sim_wq = sim_data.get('avg_wait_time_Wq', 0)
                    analytical_wq = prediction_data.get('Wq', 0)
                    
                   
                    if 'Not enough data for auto-parameterization' in prediction_data.get('error', ''):
                        st.info(
                            "Not enough data was collected during the simulation to automatically parameterize "
                            "and run the analytical model for this specific machine."
                        )
                    
                    elif abs(sim_wq) < 1e-6 and all(abs(params.get(p, 0)) < 1e-6 for p in ['lam', 'mu']):
                        st.info(
                            "No queueing activity was recorded for this machine. This typically happens if the "
                            "machine had zero utilization or received no jobs."
                        )
                   
                    else:
                        delta_str = "N/A"
                        if isinstance(analytical_wq, (int, float)) and isinstance(sim_wq, (int, float)):
                            delta_val = analytical_wq - sim_wq
                            delta_str = f"{delta_val:.3f} hrs"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("##### Simulation (Ground Truth)")
                            st.metric(
                                label="Avg. Wait in Queue (Wq)",
                                value=f"{sim_wq:.3f} hrs"
                            )
                        with col2:
                            st.write(f"##### Analytical Model ({model_type})")
                            st.metric(
                                label="Predicted Avg. Wait in Queue (Wq)",
                                value=f"{analytical_wq:.3f} hrs" if isinstance(analytical_wq, (int, float)) else "Error",
                                delta=delta_str,
                                delta_color="inverse",
                                help="The difference between the analytical prediction and the simulated reality."
                            )
                        
                        st.markdown("---")
                        st.write("###### Parameters Used for Calculation")
                        param_source = params.get("source", "N/A")
                        st.caption(f"Parameter Source: `{param_source.upper()}`")
                        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                        p_col1.metric("λ (Arrival Rate)", f"{params.get('lam', 0):.3f}")
                        p_col2.metric("μ (Service Rate)", f"{params.get('mu', 0):.3f}")
                        p_col3.metric("Ca² (Arrival SCV)", f"{params.get('arrival_scv', 0):.3f}")
                        p_col4.metric("Cs² (Service SCV)", f"{params.get('service_scv', 0):.3f}")
    
                       
                        if "error" in prediction_data and 'Not enough data' not in prediction_data['error']:
                            st.error(f"**Model Calculation Error:** {prediction_data['error']}")
                        else:
                            with st.container():
                                st.write("###### Full Analytical Results (JSON)")
                                st.json(prediction_data)
    else:
        st.warning("Analytical queueing comparison data not found in the run output.")

    st.caption(f"Source: {assembled.get('_source_dir')}")
    st.header("Advanced Financial Analysis")
    adv_financials = assembled.get('advanced_financials')
    viz_data = assembled.get('financial_viz_data')

    if adv_financials:
        st.subheader("Tier 1 & 2 Financial Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tier 1: Cost & Profitability**")
            tier1_metrics = adv_financials.get("tier1", {})
            for key, data in tier1_metrics.items():
                st.metric(label=data.get('label', key), value=f"{data.get('value', 0):.2f} {data.get('unit', '')}")
        with col2:
            st.write("**Tier 2: Health & Efficiency**")
            tier2_metrics = adv_financials.get("tier2", {})
            for key, data in tier2_metrics.items():
                st.metric(label=data.get('label', key), value=f"{data.get('value', 0):.2f} {data.get('unit', '')}")
    else:
        st.info("Advanced financial metrics were not found in the run output.")
    if viz_data:
        st.subheader("Financial Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Contribution Margin by Product**")
            contrib_data = viz_data.get('contribution_margin_by_product')
            if contrib_data:
                df_contrib = pd.DataFrame(contrib_data).set_index('product')
                st.bar_chart(df_contrib['unit_contribution_margin'])
                st.caption("Identifies the most profitable products on a per-unit basis.")
            else:
                st.warning("Contribution margin data not available.")
        with col2:
            st.write("**Downtime Costs by Machine (Pareto)**")
            pareto_data = viz_data.get('downtime_pareto')
            if pareto_data:
                df_pareto = pd.DataFrame(pareto_data).set_index('machine')
                st.bar_chart(df_pareto['cost'])
                st.caption("Highlights the machines responsible for the most downtime-related costs.")
            else:
                st.warning("Downtime cost data not available.")
    else:
        st.info("Advanced financial visualization data was not found in the run output.")
    st.subheader("SYSTEM-LEVEL ANALYSIS (Jackson Network)")
    jackson_data = assembled.get("analytical_comparison", {}).get("__jackson_network")
    if jackson_data and 'error' not in jackson_data:
        st.info("""
            This section analyzes the entire factory as an interconnected **network of queues**.
            It calculates the *effective arrival rate* (λ) at each station, which includes both
            external orders and internal transfers from other stations.
            """
        )
        total_wip = jackson_data.get('system_wip_analytical', 0)
        st.metric(
            "Predicted Average Work-In-Progress (WIP) in the System",
            value=f"{total_wip:.2f} jobs"
        )
        st.markdown("---")
        node_results = jackson_data.get('nodes', {})
        if node_results:
            st.write("##### Per-Station Performance within the Network")
            df_data = []
            for stage, data in node_results.items():
                df_data.append({
                    "Stage": stage,
                    "λ (Effective Arrival Rate)": data.get("effective_lambda", 0),
                    "μ (Service Rate)": data.get("mu", 0),
                    "c (Servers)": data.get("c", 0),
                    "ρ (Utilization)": data.get("rho", 0),
                    "Wq (Avg. Wait in Queue)": data.get("Wq", 0),
                    "L (Avg. Jobs at Station)": data.get("L", 0),
                })
            df = pd.DataFrame(df_data)
            st.dataframe(df.style
                .format({
                    "λ (Effective Arrival Rate)": "{:.3f}",
                    "μ (Service Rate)": "{:.3f}",
                    "ρ (Utilization)": "{:.2%}",
                    "Wq (Avg. Wait in Queue)": "{:.2f} hrs",
                    "L (Avg. Jobs at Station)": "{:.2f}",
                })
                .background_gradient(subset=['ρ (Utilization)'], cmap='Reds')
            )
        with st.expander("View Raw Network Parameters (Routing Matrix & Arrivals)"):
            st.write("###### External Arrival Rates (α)")
            st.json({
                stage: rate for stage, rate in zip(jackson_data.get('nodes', {}).keys(), jackson_data.get('external_arrivals_alpha', []))
            })
            st.write("###### Routing Probability Matrix (P)")
            st.write("Rows are 'From', Columns are 'To'")
            p_matrix = jackson_data.get('routing_matrix_P', [])
            stage_names = list(jackson_data.get('nodes', {}).keys())
            if p_matrix and stage_names:
                df_p = pd.DataFrame(p_matrix, index=stage_names, columns=stage_names)
                st.dataframe(df_p.style.format("{:.2%}").background_gradient(cmap='Blues'))
    elif jackson_data:
        st.error(f"Jackson Network analysis failed: **{jackson_data.get('error')}**")
    else:
        st.warning("Jackson Network analysis results not found or analysis was disabled.")
    import plotly.express as px
    st.subheader("Machine Gantt")
    bundle_viz = assemble_dashboard.__globals__.get("load_exports")(assembled.get("_source_dir"))
    df_states_raw = bundle_viz.get("machine_states.csv")
    if isinstance(df_states_raw, pd.DataFrame):
        st.write({
            "diag_raw_shape": df_states_raw.shape,
            "diag_raw_cols": list(df_states_raw.columns),
            "diag_head": df_states_raw.head(3).to_dict(orient="records")
        })
    else:
        st.write({"diag_raw": "machine_states.csv not a DataFrame"})
    sim_h = float(assembled.get("sim_time_hours") or 0.0)
    sim_h_data = None
    try:
        if isinstance(df_states_raw, pd.DataFrame) and not df_states_raw.empty:
            tmp = df_states_raw.copy()
            tmp.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in tmp.columns}, inplace=True)
            if "end" not in tmp.columns and "etime" in tmp.columns:
                tmp.rename(columns={"etime": "end"}, inplace=True)
            if "end" not in tmp.columns and "finish" in tmp.columns:
                tmp.rename(columns={"finish": "end"}, inplace=True)
            sim_h_data = pd.to_numeric(tmp.get("end", pd.Series(dtype=float)), errors="coerce").max()
    except Exception:
        sim_h_data = None
    if sim_h_data and (not sim_h or sim_h_data > sim_h):
        sim_h = float(sim_h_data)
    def normalize_states_for_gantt(df: pd.DataFrame, sim_h: float) -> pd.DataFrame:
        d = df.copy()
        d.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in d.columns}, inplace=True)
        if "state" not in d.columns and "label" in d.columns:
            d.rename(columns={"label": "state"}, inplace=True)
        st.write({
            "diag2_cols": list(d.columns)[:10],
            "has_state": "state" in d.columns,
            "sample_states": d["state"].astype(str).head(3).tolist() if "state" in d.columns else []
        })
        if "state" in d.columns:
            d["state"] = d["state"].astype(str).str.strip().str.lower().str.replace(" ", "_")
            d["state"] = d["state"].replace({
                "waiting_worker": "waitingworker",
                "waiting_worker,": "waitingworker",
                "waitingworker,": "waitingworker",
                "waiting material": "waitingmaterial",
                "waiting_material": "waitingmaterial",
                "calendar_off": "calendaroff",
                "calendar-off": "calendaroff"
            })
        if "start" not in d.columns and "stime" in d.columns:
            d.rename(columns={"stime": "start"}, inplace=True)
        if "end" not in d.columns and "etime" in d.columns:
            d.rename(columns={"etime": "end"}, inplace=True)
        if "start" not in d.columns and "begin" in d.columns:
            d.rename(columns={"begin": "start"}, inplace=True)
        if "end" not in d.columns and "finish" in d.columns:
            d.rename(columns={"finish": "end"}, inplace=True)
        if not {"start", "end"}.issubset(d.columns):
            return pd.DataFrame()
        d["start"] = pd.to_numeric(d["start"], errors="coerce")
        d["end"] = pd.to_numeric(d["end"], errors="coerce")
        d["start"] = d["start"].astype(float)
        d["end"] = d["end"].astype(float)
        d = d.dropna(subset=["start", "end"])
        d = d[d["end"] > d["start"]]
        st.write({
            "diag3_rows": len(d),
            "start_min": float(d["start"].min()),
            "start_max": float(d["start"].max()),
            "end_min": float(d["end"].min()),
            "end_max": float(d["end"].max())
        })
        if d.empty:
            return d
        q95 = pd.concat([d["start"], d["end"]]).quantile(0.95)
        factor = 1.0
        if sim_h and sim_h > 0 and pd.notna(q95) and q95 > (10.0 * sim_h):
            factor = 1.0 / 3600.0
        d["start"] = d["start"] * factor
        d["end"] = d["end"] * factor
        st.write({"q95": float(q95) if pd.notna(q95) else None, "factor": factor, "rows_after_unit": len(d)})
        if (not sim_h) or sim_h <= 0:
            sim_h = float(d["end"].max()) if len(d) else 0.0
        if sim_h and sim_h > 0:
            d["end"] = d["end"].clip(upper=sim_h)
            d = d[d["end"] > d["start"]]
        st.write({"sim_h_used": sim_h, "rows_after_clip": len(d), "end_max_after_clip": float(d["end"].max()) if len(d) else None})
        if "state" in d.columns:
            d["state"] = d["state"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        if "stage" in d.columns and "machine" in d.columns:
            d["resource"] = d["stage"].astype(str) + "/" + d["machine"].astype(str)
            y_col = "resource"
        elif "machine" in d.columns:
            y_col = "machine"
        elif "stage" in d.columns:
            y_col = "stage"
        else:
            d["resource"] = "resource"
            y_col = "resource"
        st.write({"y_col": y_col, "has_y_col": (y_col in d.columns) if y_col else False, "cols_now": list(d.columns)})
        cols = ["start", "end"]
        if "state" in d.columns:
            cols.append("state")
        if y_col in d.columns:
            cols.append(y_col)
        d = d[cols].copy()
        return d
    if isinstance(df_states_raw, pd.DataFrame) and not df_states_raw.empty:
        dfg = normalize_states_for_gantt(df_states_raw, sim_h)
        if dfg.empty:
            st.warning("No valid intervals for Gantt after normalization. Check machine_states.csv columns and time units.")
            st.dataframe(df_states_raw.head(10))
        else:
            y_col = "resource" if "resource" in dfg.columns else ("machine" if "machine" in dfg.columns else "stage")
            dfg[y_col] = pd.Categorical(dfg[y_col], categories=sorted(dfg[y_col].unique()), ordered=True)
            mask = (dfg["end"] - dfg["start"]) <= 0
            dfg.loc[mask, "end"] = dfg.loc[mask, "start"] + 1e-6
            dfg_plot = dfg.copy()
            dfg_plot["start"] = dfg_plot["start"].astype(float)
            dfg_plot["end"] = dfg_plot["end"].astype(float)
            dfg_plot["duration"] = (dfg_plot["end"] - dfg_plot["start"]).clip(lower=1e-6)
            dfg_plot[y_col] = pd.Categorical(dfg_plot[y_col], categories=sorted(dfg_plot[y_col].unique()), ordered=True)
            import plotly.express as px
            fig = px.bar(
                dfg_plot,
                x="duration",
                y=y_col,
                color="state" if "state" in dfg_plot.columns else None,
                orientation="h",
                base="start",  
                title="Machine State Timeline (hours)"
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                height=500,
                legend_title_text="State",
                xaxis_title="Sim time (hours)",
                xaxis=dict(type="linear", range=[0, sim_h] if sim_h and sim_h > 0 else None),
                margin=dict(l=20, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    else:
        st.info("No machine_states.csv available for Gantt.")



    st.subheader("Order Flow (Sankey)")
    df_orders = bundle_viz.get("orders_summary.csv")
    df_fg = bundle_viz.get("finished_goods.csv")
    df_ship = bundle_viz.get("shipped_goods.csv")
    
    
    def get_item_counts_from_orders(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0, 0
        total_demand = pd.to_numeric(df.get("items_total", 0), errors="coerce").fillna(0).sum()
        total_completed = pd.to_numeric(df.get("items_completed", 0), errors="coerce").fillna(0).sum()
        return int(total_demand), int(total_completed)
    
   
    total_demand, total_completed = get_item_counts_from_orders(df_orders)
    items_shipped = len(df_ship) if isinstance(df_ship, pd.DataFrame) else 0
    
  
    backlog = max(0, total_demand - total_completed)
    in_finished_goods = max(0, total_completed - items_shipped)
    
    import plotly.graph_objects as go
    
   
    labels = [
        f"Total Demand ({total_demand})",         # Node 0
        f"Completed ({total_completed})",         # Node 1
        f"Unfulfilled Backlog ({backlog})",      # Node 2
        f"Shipped ({items_shipped})",             # Node 3
        f"In Finished Goods Inv. ({in_finished_goods})" # Node 4
    ]
    source_nodes = [0, 0, 1, 1]  # From: Demand, Demand, Completed, Completed
    target_nodes = [1, 2, 3, 4]  # To:   Completed, Backlog, Shipped, Finished Goods
    flow_values = [
        total_completed,
        backlog,
        items_shipped,
        in_finished_goods
    ]
    
    
    flow_values = [max(v, 0.0001) for v in flow_values]
    
    
    colors = [
        '#3b82f6', # Blue for Demand
        '#22c55e', # Green for Completed
        '#f97316', # Orange for Backlog
        '#8b5cf6', # Purple for Shipped
        '#f43f5e'  # Rose for Inventory
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=source_nodes,
            target=target_nodes,
            value=flow_values
        )
    )])
    
    fig.update_layout(
        title_text="Order Fulfillment Flow (by Item Count)",
        font_size=12,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

    st.subheader("Queue and WIP Distribution")
    df_states_q = bundle_viz.get("machine_states.csv")
    sim_h = float(assembled.get("sim_time_hours") or 0.0)
    sim_h_data = None
    try:
        if isinstance(df_states_raw, pd.DataFrame) and not df_states_raw.empty:
            tmp = df_states_raw.copy()
            tmp.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in tmp.columns}, inplace=True)
            if "end" not in tmp.columns and "etime" in tmp.columns:
                tmp.rename(columns={"etime": "end"}, inplace=True)
            if "end" not in tmp.columns and "finish" in tmp.columns:
                tmp.rename(columns={"finish": "end"}, inplace=True)
            sim_h_data = pd.to_numeric(tmp.get("end", pd.Series(dtype=float)), errors="coerce").max()
    except Exception:
        sim_h_data = None
    if sim_h_data and (not sim_h or sim_h_data > sim_h):
        sim_h = float(sim_h_data)
    def normalize_states(df, sim_h):
        df = df.copy()
        df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns}, inplace=True)
        if "state" not in df.columns and "label" in df.columns:
            df.rename(columns={"label": "state"}, inplace=True)
        if not {"start","end"}.issubset(df.columns):
            return pd.DataFrame()
        s = pd.to_numeric(df["start"], errors="coerce")
        e = pd.to_numeric(df["end"], errors="coerce")
        factor = 1.0
        try:
            mx = pd.concat([s, e]).max()
            if sim_h and sim_h > 0 and mx and mx > (10.0 * sim_h):
                factor = 1.0 / 3600.0
        except Exception:
            pass
        s = s * factor
        e = e * factor
        if sim_h and sim_h > 0:
            s = s.clip(lower=0, upper=sim_h)
            e = e.clip(lower=0, upper=sim_h)
        df["start"] = s
        df["end"] = e
        df = df[df["end"] > df["start"]]
        df["duration"] = (df["end"] - df["start"]).astype(float)
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        return df
    if isinstance(df_states_q, pd.DataFrame) and not df_states_q.empty:
        dfx = normalize_states(df_states_q, sim_h)
        if dfx.empty:
            st.info("States data invalid for WIP charts.")
        else:
            non_states = ["idle","waitingworker","waitingmaterial","setup"]
            nonrun = dfx[dfx["state"].isin(non_states)]
            agg = nonrun.groupby("state", as_index=False)["duration"].sum()
            if "machine" in dfx.columns:
                n_machines = dfx["machine"].astype(str).nunique()
            elif "resource" in dfx.columns:
                n_machines = dfx["resource"].astype(str).nunique()
            else:
                n_machines = 1
            total_machine_hours = (sim_h * n_machines) if sim_h and sim_h > 0 else None
            if total_machine_hours:
                agg["share_percent"] = agg["duration"] / total_machine_hours * 100.0
            import plotly.express as px
            fig3 = px.bar(agg, x="state", y="duration", title="Non-running Time by State (machine-hours)")
            st.plotly_chart(fig3, use_container_width=True)
            if total_machine_hours:
                fig3b = px.bar(agg, x="state", y="share_percent", title="Non-running Share by State (% of machine capacity)")
                st.plotly_chart(fig3b, use_container_width=True)
    else:
        st.info("No machine_states.csv available for WIP distribution.")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Industrial Dashboard")
    parser.add_argument("--output_dir", required=True, help="Path to simulation output directory")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server at :8000")
    parser.add_argument("--ui", action="store_true", help="Start Streamlit UI instead of API")
    args = parser.parse_args()
    if args.ui:
        if not HAS_STREAMLIT:
            print("Streamlit not installed. Run: pip install streamlit")
            sys.exit(1)
        run_streamlit(args.output_dir)
    elif args.serve:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            obj = assemble_dashboard(resolve_latest_run(args.output_dir))
            print(json.dumps(obj, indent=2))
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr)
            sys.exit(1)