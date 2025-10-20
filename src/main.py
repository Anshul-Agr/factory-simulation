#STANDARD IMPORTS

import csv
import json
import math
import os
import random
import time
from typing import Dict, List
from pathlib import Path


# 3rd PARTY IMPORTS

import matplotlib.pyplot as plt
import numpy as np
import simpy

#LOCAL IMPORTS

from config import (
    ANALYTICAL_QUEUE_CONFIG, CALENDAR_CONFIG, CHART_CONFIG, CSV_ORDERS_PATH,
    DAILY_ORDERS, DEFAULT_MACHINE_CONFIG, ENVIRONMENTCONFIG, FINANCIAL_CONFIG,
    INVENTORY_POLICY, MACHINE_DEFINITIONS, MACHINE_ENERGY_PROFILES,
    MACHINE_WORKER_REQUIREMENTS, MATERIAL_CONSUMPTION, MATERIAL_ENV,
    PLANNING_CONFIG, PRODUCT_ENV, PRODUCTS, QUALITYCONFIG,
    QUEUING_CONFIG, RANDOM_SEED, RAW_MATERIALS, SHIPPING_CONFIG,
    SIMULATION_TIME, TRANSPORTCONFIG, TRANSPORT_ENV, USE_CSV_ORDERS,
    WORKER_DEFINITIONS, WORKER_POOL_CONFIG, WORKER_SCHEDULING,
    WORKER_SKILLS, WORKER_UNPREDICTABILITY
)
from financial_analysis import FinancialAnalysis
from forecasting import DemandGenerator, Forecaster
from inventory_utils import abc_classification, calculate_eoq
from logging_export import (JobLogger, MetricsBus, export_all,
                            load_orders_from_csv)
from metrics import AlertsManager, BottleneckDetector, ComprehensiveKPIs
from queuing_system import (ANALYTICAL_MODELS, Job, PolicyQueue,
                            get_policy_from_config)



rt_metrics = MetricsBus()
job_logger = JobLogger()
print("="*60)
print("FACTORY SIMULATION CONFIGURATION LOADED")
print("="*60)
print(f"Products defined: {len(PRODUCTS)}")
print(f"Machine types: {len(MACHINE_DEFINITIONS)}")
print(f"Order batches: {len(DAILY_ORDERS)}")
print(f"Simulation duration: {SIMULATION_TIME} hours")
print("="*60)

def pick_machine(machines_for_stage):
    if isinstance(machines_for_stage, list):
        return min(machines_for_stage, key=lambda m: (len(m.queue), getattr(m, 'name', '')))
    return machines_for_stage
class ShiftCalendar:
    def __init__(self, config):
        self.cfg = config
        self.week_len = config.get('week_length', 168)
    def _day_hour(self, t):
        t_wrapped = t % self.week_len
        day = int(t_wrapped // 24)
        hour = t_wrapped % 24
        return day, hour
    def _in_break(self, hour, breaks):
        for bstart, bend in breaks:
            if bstart <= hour < bend:
                return True
            if bend > 24 and (hour < (bend - 24)):
                return True
        return False
    def _in_shift(self, hour, shift):
        s, e = shift['start'], shift['end']
        if s <= e:
            return s <= hour < e
        else:
            return (s <= hour < 24) or (0 <= hour < (e - 24))
    def _get_stage_overrides(self, stage_name):
        daily_shifts = self.cfg.get('daily_shifts', [])
        working_days = self.cfg.get('working_days', list(range(7)))
        if stage_name and stage_name in self.cfg.get('machine_overrides', {}):
            ov = self.cfg['machine_overrides'][stage_name]
            daily_shifts = ov.get('daily_shifts', daily_shifts)
            working_days = ov.get('working_days', working_days)
        return daily_shifts, working_days
    def is_working_time(self, t, stage_name=None):
        day, hour = self._day_hour(t)
        for h in self.cfg.get('holidays', []):
            if abs(t - h) < 1e-9:
                return False
        if stage_name is None:
            all_stage_configs = list(self.cfg.get('machine_overrides', {}).keys())
            all_stage_configs.append(None) 
            for stage in all_stage_configs:
                daily_shifts, working_days = self._get_stage_overrides(stage)
                if day in working_days:
                    for sh in daily_shifts:
                        if self._in_shift(hour, sh) and not self._in_break(hour, sh.get('breaks', [])):
                            return True 
            return False 
        daily_shifts, working_days = self._get_stage_overrides(stage_name)
        if day not in working_days:
            return False
        for sh in daily_shifts:
            if self._in_shift(hour, sh):
                if self._in_break(hour, sh.get('breaks', [])):
                    return False
                return True
        return False
    def next_working_time(self, t, stage_name=None):
        step = 0.05  
        cur = t
        guard = int(self.week_len / step) + 1000
        for _ in range(guard):
            if self.is_working_time(cur, stage_name=stage_name):
                return cur
            cur += step
        return t
def calendar_pause_until_working(env, stage_name, calendar, timeline):
    if not calendar.is_working_time(env.now, stage_name=stage_name):
        t1 = calendar.next_working_time(env.now, stage_name=stage_name)
        if t1 > env.now:
            if timeline:
                timeline.add_interval(env.now, t1, 'calendar_off')
                timeline.set_state(t1, 'idle')
            yield env.timeout(t1 - env.now)
def calendar_timeout(env, duration, stage_name, calendar, timeline):
    remaining = float(duration)
    eps = 1e-6          
    min_step = 1.0/240  
    max_step = 0.25     
    while remaining > eps:
        if not calendar.is_working_time(env.now, stage_name=stage_name):
            t1 = calendar.next_working_time(env.now, stage_name=stage_name)
            if t1 > env.now + eps:
                timeline.add_interval(env.now, t1, 'calendar_off')
                yield env.timeout(t1 - env.now)
            else:
                yield env.timeout(min_step)
            continue
        step = min(remaining, max_step)
        end_chunk = env.now + step
        guard = 0
        while step > min_step and not calendar.is_working_time(end_chunk, stage_name=stage_name):
            step *= 0.5
            end_chunk = env.now + step
            guard += 1
            if guard > 30:  
                break
        if not calendar.is_working_time(end_chunk, stage_name=stage_name):
            step = min_step
            end_chunk = env.now + step
        start_chunk = env.now
        yield env.timeout(step)
        elapsed = env.now - start_chunk
        if elapsed <= 0:
            yield env.timeout(min_step)
            elapsed = min_step
        remaining -= elapsed
class StateTimeline:
    def __init__(self, machine_name):
        self.machine_name = machine_name
        self.intervals = []
        self._current_label = None
        self._current_start = None
    def set_state(self, t, label):
        if self._current_label is not None and self._current_start is not None and t > self._current_start:
            self.intervals.append((self._current_start, t, self._current_label))
        self._current_label = label
        self._current_start = t
    def end_all(self, t_end):
        if self._current_label is not None and self._current_start is not None and t_end > self._current_start:
            self.intervals.append((self._current_start, t_end, self._current_label))
        self._current_label = None
        self._current_start = None
    def add_interval(self, t0, t1, label):
        if t1 > t0:
            self.intervals.append((t0, t1, label))
    def compress(self):
        if not self.intervals:
            return
        self.intervals.sort(key=lambda x: x[0])
        merged = []
        s, e, lab = self.intervals[0]
        for i in range(1, len(self.intervals)):
            s2, e2, lab2 = self.intervals[i]
            if lab2 == lab and abs(s2 - e) < 1e-9:
                e = e2
            else:
                merged.append((s, e, lab))
                s, e, lab = s2, e2, lab2
        merged.append((s, e, lab))
        self.intervals = merged
    def totals(self):
        agg = {}
        for s, e, lab in self.intervals:
            agg[lab] = agg.get(lab, 0.0) + (e - s)
        return agg
    def get_as_dicts(self):
        if not hasattr(self, 'machine_name'):
            self.machine_name = "Unknown"
        return [
            {'machine': self.machine_name, 'start': s, 'end': e, 'label': l}
            for s, e, l in self.intervals
        ]
def compute_oee_from_timeline(timeline: StateTimeline, horizon_end, availability_counts_pm=True,
                              good_units=None, total_units=None,
                              observed_run_time=None, ideal_run_time=None):
    timeline.compress()
    totals = timeline.totals()
    total_time = horizon_end
    cal_off = totals.get('calendar_off', 0.0)
    ppt = max(0.0, total_time - cal_off)
    unplanned_dt = totals.get('breakdown', 0.0)
    pm_dt = totals.get('maintenance', 0.0)
    dt_for_avail = (unplanned_dt + pm_dt) if availability_counts_pm else unplanned_dt
    running_time = totals.get('running', 0.0)
    setup_time = totals.get('setup', 0.0)
    waits = totals.get('waiting_material', 0.0) + totals.get('waiting_worker', 0.0)
    available_time = max(0.0, ppt - dt_for_avail)
    availability = (available_time / ppt) if ppt > 0 else 0.0
    if ideal_run_time is None:
        performance = 1.0
    else:
        if observed_run_time is None:
            observed_run_time = running_time
        performance = (observed_run_time / ideal_run_time) if ideal_run_time > 0 else 1.0
        performance = max(0.0, min(1.5, performance))
    if good_units is None or total_units is None or total_units == 0:
        quality = 1.0
    else:
        quality = good_units / total_units
    oee = availability * performance * quality
    return {
        'total_time': total_time, 'calendar_off': cal_off, 'ppt': ppt,
        'unplanned_dt': unplanned_dt, 'pm_dt': pm_dt, 'dt_for_avail': dt_for_avail,
        'available_time': available_time, 'running_time': running_time,
        'setup_time': setup_time, 'wait_time': waits,
        'availability': availability, 'performance': performance,
        'quality': quality, 'oee': oee, 'label_totals': totals
    }
calendar = ShiftCalendar(CALENDAR_CONFIG)
class EnvironmentalTracker:
    def __init__(self, env, MACHINE_DEFINITIONS, ENVIRONMENTCONFIG, energy_profiles,transport_env=None, product_env=None, material_env=None):
        self.env = env
        self.mdefs = MACHINE_DEFINITIONS
        self.cfg = ENVIRONMENTCONFIG
        self.energy_profiles = energy_profiles
        self.transport_env = transport_env or {}
        self.product_env = product_env or {}
        self.material_env = material_env or {}
        self.machine_energy_kwh = {}         
        self.machine_emissions_kg = {}       
        self.machine_state_energy = []       
        self.transport_events = []           
        self.material_events = []            
        self.product_emissions_kg = {}       
        self.product_energy_kwh = {}         
        self.product_units_completed = {}    
        self.waste_generated_kg = 0.0
        self.waste_recycled_kg = 0.0
        self.water_used_m3 = 0.0
        self.total_energy_kwh = 0.0
        self.total_emissions_kg = 0.0
    def _grid_emission(self, kwh):
        return kwh * self.cfg.get("grid_emission_factor_kg_per_kwh", 0.7)
    def _diesel_emission_for_km(self, km, l_per_km):
        ef = self.cfg.get("diesel_emission_factor_kg_per_l", 2.68)
        return km * l_per_km * ef
    def _electric_emission_for_km(self, km, kwh_per_km):
        return self._grid_emission(km * kwh_per_km)
    def record_machine_state_block(self, machine_name, stage_name, label, start, end):
        if end <= start:
            return
        prof = self.energy_profiles.get(stage_name, {})
        map_label = {"waiting_material": "waitingmaterial", "waiting_worker": "waitingworker", "calendar_off": "calendaroff"}.get(label, label)
        if map_label == "calendaroff":
            kw = prof.get("calendaroff_kW", 0.0)
        else:
            kw = prof.get(f"{map_label}_kW", prof.get("idle_kW", 0.0))
        hours = max(0.0, end - start)
        kwh = kw * hours
        co2 = self._grid_emission(kwh)
        self.total_energy_kwh += kwh
        self.total_emissions_kg += co2
        self.machine_energy_kwh[machine_name] = self.machine_energy_kwh.get(machine_name, 0.0) + kwh
        self.machine_emissions_kg[machine_name] = self.machine_emissions_kg.get(machine_name, 0.0) + co2
        self.machine_state_energy.append({
            "machine": machine_name, "stage": stage_name, "state": label,
            "start": start, "end": end, "hours": hours, "kwh": kwh, "co2_kg": co2
        })
    def record_transport(self, mode, km, hours=0.0):
        row = {"sim_time": self.env.now if hasattr(self, "env") else 0.0,
               "mode": mode, "km": km, "hours": hours}
        energy_mode = self.transport_env.get(mode, {}).get("energy_mode", "electric")
        if energy_mode == "diesel":
            lpkm = self.transport_env.get(mode, {}).get("l_per_km", 0.0)
            co2 = self._diesel_emission_for_km(km, lpkm)
            kwh_equiv = km * lpkm * self.cfg.get("kwh_per_l_diesel_equiv", 9.7)
        else:
            kpkm = self.transport_env.get(mode, {}).get("kwh_per_km", 0.0)
            kwh_equiv = km * kpkm
            co2 = self._electric_emission_for_km(km, kpkm)
        self.total_energy_kwh += kwh_equiv
        self.total_emissions_kg += co2
        row.update({"kwh": kwh_equiv, "co2_kg": co2})
        self.transport_events.append(row)
    def record_material_consumption(self, material, quantity, product=None, stage=None):
        env_f = MATERIAL_ENV.get(material, {})
        waste_factor = float(env_f.get("waste_factor", 0.0))
        water_m3 = float(env_f.get("water_m3_per_unit", 0.0)) * float(quantity)
        waste_kg = waste_factor * float(quantity)
        recycle_rate = self.product_env.get(product, {}).get("recycle_rate", 0.0) if product else 0.0
        recycled_kg = waste_kg * recycle_rate
        self.waste_generated_kg += waste_kg
        self.waste_recycled_kg += recycled_kg
        self.water_used_m3 += water_m3
        self.material_events.append({
            "sim_time": self.env.now if hasattr(self, "env") else 0.0,
            "type": "consumption", "material": material, "quantity": quantity,
            "product": product, "stage": stage, "waste_kg": waste_kg,
            "recycled_kg": recycled_kg, "water_m3": water_m3
        })
    def record_material_delivery(self, material, quantity):
        self.material_events.append({
            "sim_time": self.env.now if hasattr(self, "env") else 0.0,
            "type": "delivery", "material": material, "quantity": quantity
        })
    def record_product_completion(self, product, cycletime_hours, machines_used):
        self.product_units_completed[product] = self.product_units_completed.get(product, 0) + 1
        waste_kg = self.product_env.get(product, {}).get("waste_kg", 0.0)
        recycle_rate = self.product_env.get(product, {}).get("recycle_rate", 0.0)
        recycled_kg = waste_kg * recycle_rate
        self.waste_generated_kg += waste_kg
        self.waste_recycled_kg += recycled_kg
        if self.cfg.get("enable_water_tracking", True):
            self.water_used_m3 += self.product_env.get(product, {}).get("water_m3", 0.0)
    def compute_kpis(self, simulation_time_hours):
        units_total = sum(self.product_units_completed.values()) or 1
        kpi = {
            "total_energy_kwh": self.total_energy_kwh,
            "total_emissions_kg": self.total_emissions_kg,
            "total_waste_kg": self.waste_generated_kg,
            "waste_recycled_kg": self.waste_recycled_kg,
            "water_used_m3": self.water_used_m3,
            "energy_per_hour_kwh": (self.total_energy_kwh / simulation_time_hours) if simulation_time_hours > 0 else 0.0,
            "emissions_per_hour_kg": (self.total_emissions_kg / simulation_time_hours) if simulation_time_hours > 0 else 0.0,
            "emissions_per_unit_kg": (self.total_emissions_kg / units_total),
            "energy_per_unit_kwh": (self.total_energy_kwh / units_total),
            "per_machine_energy_kwh": self.machine_energy_kwh,
            "per_machine_emissions_kg": self.machine_emissions_kg,
            "per_product_units": self.product_units_completed
        }
        return kpi
class OrderTracker:
    def __init__(self, env):
        self.env = env
        self.orders = {}
        self.events = []
    def register_order(self, order_id, products, release_time, due_in_hours=None):
        if order_id in self.orders:
            print(f"WARNING: Order ID {order_id} already exists. Skipping registration.")
            return
        items = [{'item_id': f"{order_id}_item_{idx}", 'product': p, 'status': 'pending', 'job_id': None, 'start': None, 'finish': None} for idx, p in enumerate(products, start=1)]
        self.orders[order_id] = {
            'order_id': order_id, 'release_time': release_time,
            'due_time': (release_time + due_in_hours) if due_in_hours else None,
            'items': items, 'start_time': None, 'completion_time': None,
            'shipped_time': None, 'status': 'open' 
        }
        self.events.append({'time': self.env.now, 'order_id': order_id, 'event': 'released', 'count': len(items)})
    def mark_item_started(self, order_id, item_id, job_id):
        if order_id not in self.orders: return
        order = self.orders[order_id]
        if order['status'] == 'open':
            order['status'] = 'in_progress'
            order['start_time'] = self.env.now
            self.events.append({'time': self.env.now, 'order_id': order_id, 'event': 'started'})
        for item in order['items']:
            if item['item_id'] == item_id:
                item['status'] = 'in_progress'
                item['job_id'] = job_id
                item['start'] = self.env.now
                break
    def mark_item_completed(self, order_id, item_id):
        if order_id not in self.orders: return
        order = self.orders[order_id]
        item_completed = False
        for item in order['items']:
            if item['item_id'] == item_id:
                item['status'] = 'completed'
                item['finish'] = self.env.now
                item_completed = True
                break
        if item_completed:
            all_items_completed = all(it['status'] == 'completed' for it in order['items'])
            if all_items_completed:
                order['status'] = 'completed'
                order['completion_time'] = self.env.now
                self.events.append({'time': self.env.now, 'order_id': order_id, 'event': 'completed'})
                print(f"ORDER COMPLETE: Order {order_id} is now fully produced and ready for shipment.")
    def mark_order_shipped(self, order_id):
        if order_id not in self.orders: return
        order = self.orders[order_id]
        if order['status'] == 'completed':
            order['status'] = 'shipped'
            order['shipped_time'] = self.env.now
            self.events.append({'time': self.env.now, 'order_id': order_id, 'event': 'shipped'})
    def summary(self):
        summary = {'open': [], 'in_progress': [], 'completed': [], 'shipped': []}
        for oid, order in self.orders.items():
            if order['status'] in summary:
                summary[order['status']].append(oid)
        return summary
    def order_progress(self, order_id):
        if order_id not in self.orders: return 0, 0
        items = self.orders[order_id]['items']
        done = sum(1 for it in items if it['status'] == 'completed')
        return done, len(items)
    def lateness(self, order_id):
        if order_id not in self.orders: return None
        o = self.orders[order_id]
        if o['due_time'] is not None and o['completion_time'] is not None:
            return max(0, o['completion_time'] - o['due_time'])
        return None
class FinancialTracker:
    def __init__(self, env, MACHINE_DEFINITIONS, FINANCIAL_CONFIG, PRODUCTS, environmental=None):
        self.env = env
        self.machine_definitions = MACHINE_DEFINITIONS
        self.config = FINANCIAL_CONFIG
        self.environmental = environmental
        self.PRODUCTS = PRODUCTS 
        self.revenue_events = []
        self.downtime_losses = []
        self.procurement_costs = []
        self.labor_cost_per_hour = self.config.get('labor_cost_per_hour', 20.0)
        self.downtime_cost_per_hour = self.config.get('downtime_cost_per_hour', 50.0)
        self.total_revenue = 0.0
        self.total_costs = 0.0
        self.total_downtime_cost = 0.0
        self.total_procurement_cost = 0.0
        self.stockout_costs = []
        self.total_stockout_cost = 0.0
        self.total_machine_operating_cost = 0.0
        self.stockout_cost_per_unit = self.config.get('stockout_cost_per_unit', 15.0)
    def get_machine_config(self, machine_name):
        for stage, machines in self.machine_definitions.items():
            if isinstance(machines, list):
                for i, m in enumerate(machines):
                    if f"{stage}_Machine{i+1}" == machine_name:
                        return self.machine_definitions[stage]
            else:
                if f"{stage}_Machine" == machine_name:
                    return self.machine_definitions[stage]
        for stage in self.machine_definitions:
            if stage in machine_name:
                return self.machine_definitions[stage]
        return {} 
    def record_transport_cost(self, env, mode, time_hours, distance_km, operating_cost_per_hour, energy_cost_per_km):
        total = time_hours * operating_cost_per_hour + distance_km * energy_cost_per_km
        if not hasattr(self, "costevents"):
            self.costevents = []
        self.costevents.append({
            "sim_time": env.now,
            "type": "transport",
            "mode": mode,
            "hours": time_hours,
            "km": distance_km,
            "cost": total
        })
        if hasattr(self, "totalcosts"):
            self.totalcosts += total
        else:
            self.totalcosts = total
    def record_production_completion(self, job_id, product_type, completion_time, cycle_time, machines_used):
        if product_type not in self.PRODUCTS:
            print(f"FATAL: FinancialTracker could not find product '{product_type}' in its configuration.")
            return
        product_config = self.PRODUCTS[product_type]
        price = product_config.get('unit_price', 0)
        total_machine_cost = 0
        for machine_name in machines_used:
            config = self.get_machine_config(machine_name)
            total_machine_cost += cycle_time * config.get('operating_cost_per_hour', 0)
        self.total_machine_operating_cost += total_machine_cost
        
        total_production_cost = total_machine_cost
        net_profit = price - total_production_cost
        self.revenue_events.append({
            'time': completion_time,
            'job_id': job_id,
            'product_type': product_type,
            'gross_profit': price,
            'production_cost': total_production_cost,
            'machine_cost': total_machine_cost,
            'labor_cost': 0,  
            'net_profit': net_profit,
            'cycle_time': cycle_time
        })
        self.total_revenue += price
        self.total_costs += total_production_cost
        if hasattr(self, "environmental") and self.environmental:
            self.environmental.record_product_completion(product_type, cycle_time, machines_used)
    def record_downtime_cost(self, machine_name, event_type, start_time, duration):
        config = self.get_machine_config(machine_name)
        maintenance_cost = config.get('maintenance_cost', 0)
        if event_type == 'breakdown':
            repair_cost = maintenance_cost * config.get('breakdown_repair_multiplier', 1.5)
            downtime_cost = duration * self.downtime_cost_per_hour
        else: 
            repair_cost = maintenance_cost
            downtime_cost = duration * (self.downtime_cost_per_hour * 0.3)
        total_downtime_cost = repair_cost + downtime_cost
        self.downtime_losses.append({
            'time': start_time, 'machine': machine_name, 'event_type': event_type,
            'duration': duration, 'repair_cost': repair_cost,
            'opportunity_cost': downtime_cost, 'total_cost': total_downtime_cost
        })
        self.total_downtime_cost += total_downtime_cost
        self.total_costs += total_downtime_cost
    def record_procurement_cost(self, material, cost, quantity):
        self.procurement_costs.append({
            'material': material,
            'cost': cost,
            'quantity': quantity,
            'time': self.env.now if hasattr(self, 'env') else 0
        })
        self.total_procurement_cost += cost
        self.total_costs += cost
        print(f"COST: Recorded procurement cost ${cost:.2f} for {quantity:.2f} {material}")
    def record_stockout_cost(self, material, shortage_qty):
        cost = shortage_qty * self.stockout_cost_per_unit
        self.stockout_costs.append({
            'time': self.env.now,
            'material': material,
            'quantity': shortage_qty,
            'cost': cost
        })
        self.total_stockout_cost += cost
        self.total_costs += cost
        print(f"COST: Recorded stockout cost ${cost:.2f} for {shortage_qty:.2f} units of {material}")
    def get_financial_summary(self, simulation_time,final=False):
        net_profit = self.total_revenue - self.total_costs
        profit_margin = (net_profit / self.total_revenue * 100) if self.total_revenue > 0 else 0
        revenue_per_hour = self.total_revenue / simulation_time if simulation_time > 0 else 0
        profit_per_hour = net_profit / simulation_time if simulation_time > 0 else 0
        product_performance = {}
        for event in self.revenue_events:
            ptype = event['product_type']
            if ptype not in product_performance:
                product_performance[ptype] = {
                    'units': 0, 'revenue': 0, 'cost': 0, 'net_profit': 0,
                    'avg_cycle_time': 0, 'total_cycle_time': 0
                }
            product_performance[ptype]['units'] += 1
            product_performance[ptype]['revenue'] += event['gross_profit']
            product_performance[ptype]['cost'] += event['production_cost']
            product_performance[ptype]['net_profit'] += event['net_profit']
            product_performance[ptype]['total_cycle_time'] += event['cycle_time']
        for ptype in product_performance:
            perf = product_performance[ptype]
            if perf['units'] > 0:
                perf['avg_cycle_time'] = perf['total_cycle_time'] / perf['units']
                perf['profit_per_unit'] = perf['net_profit'] / perf['units']
            else:
                perf['avg_cycle_time'] = 0
                perf['profit_per_unit'] = 0
        return {
            'total_revenue': self.total_revenue, 'total_costs': self.total_costs,
            'net_profit': net_profit,
            'profit_margin': profit_margin, 'revenue_per_hour': revenue_per_hour,
            'profit_per_hour': profit_per_hour, 'product_performance': product_performance,
            'simulation_time': simulation_time,
            'total_procurement_cost': self.total_procurement_cost,
            'total_machine_operating_cost': self.total_machine_operating_cost,
            'total_downtime_cost': self.total_downtime_cost,
            'total_stockout_cost': self.total_stockout_cost
        }
    def get_total_labor_cost(self):
        total_production_cost = sum(e.get('production_cost', 0) for e in self.revenue_events)
        total_downtime_cost = sum(e.get('total_cost', 0) for e in self.downtime_losses)
        total_procurement_cost = sum(e.get('cost', 0) for e in self.procurement_costs)
        total_stockout_cost = sum(e.get('cost', 0) for e in self.stockout_costs)
        other_costs = (
            total_production_cost +
            total_downtime_cost +
            total_procurement_cost +
            total_stockout_cost
        )
        labor_cost = self.total_costs - other_costs
        return max(0, labor_cost)
class QualityTracker:
    def __init__(self, env, qcfg, financial=None, environmental=None):
        self.env = env
        self.cfg = qcfg
        self.financial = financial
        self.environmental = environmental
        self.stage_counts = {}  
        self.product_counts = {}  
        self.escapes_by_product = {}  
        self.shipped_good = {}  
        self.shipped_suspect = {}  
        self.returns_by_product = {}  
        self.quality_events = []  
        self.return_events = []  
        self.order_satisfaction = {}  
    def _get(self, dct, key, init):
        if key not in dct:
            dct[key] = init() if callable(init) else init.copy() if isinstance(init, dict) else init
        return dct[key]
    def _defect_p(self, product, stage):
        prod_map = self.cfg.get("defect_p", {})
        prod_over = prod_map.get(product, {})
        if stage in prod_over:
            return float(prod_over[stage])
        default_map = prod_map.get("default", {})
        return float(default_map.get(stage, 0.0))
    def _detect_p(self, qc_present):
        d = self.cfg.get("detect_p", {"with_qc": 0.9, "no_qc": 0.6})
        return float(d["with_qc"] if qc_present else d["no_qc"])
    def record_stage_result(self, stage, product, duration_h, qc_present, detected_defect, reworked, scrap, escaped):
        row = {
            "sim_time": self.env.now if hasattr(self, "env") else 0.0,
            "product": product,
            "stage": stage,
            "duration_h": duration_h,
            "qc_present": bool(qc_present),
            "defect_detected": int(detected_defect),
            "reworked": int(reworked),
            "scrap": int(scrap),
            "escaped": int(escaped),
        }
        self.quality_events.append(row)
        sc = self._get(self.stage_counts, (product, stage), lambda: {
            "produced": 0, "inspected": 0, "defects_detected": 0,
            "rework": 0, "scrap": 0, "escapes": 0
        })
        sc["produced"] += 1
        sc["inspected"] += 1  
        if detected_defect:
            sc["defects_detected"] += 1
        if reworked:
            sc["rework"] += 1
        if scrap:
            sc["scrap"] += 1
        if escaped:
            sc["escapes"] += 1
        pc = self._get(self.product_counts, product, lambda: {
            "produced": 0, "inspected": 0, "defects_detected": 0,
            "rework": 0, "scrap": 0, "escapes": 0
        })
        for k in ["produced", "inspected", "defects_detected", "rework", "scrap", "escapes"]:
            pc[k] += (1 if k in ["produced", "inspected"] else row["defect_detected"] if k == "defects_detected" else row.get(k, 0))
    def estimate_defect_outcome(self, product, stage, qc_present):
        import random
        defect_p = self._defect_p(product, stage)
        occurs = random.random() < defect_p
        if not occurs:
            return {"occurs": False, "detected": False}
        detect_p = self._detect_p(qc_present)
        detected = random.random() < detect_p
        return {"occurs": True, "detected": detected}
    def rework_decision(self):
        rcfg = self.cfg.get("rework", {"enabled": True, "max_loops": 1, "time_factor": 0.5, "success_p": 0.7})
        return rcfg["enabled"], int(rcfg["max_loops"]), float(rcfg["time_factor"]), float(rcfg["success_p"])
    def scrap_cost_for(self, product):
        m = self.cfg.get("scrap_cost_per_unit", {"default": 10.0})
        return float(m.get(product, m.get("default", 0.0)))
    def record_escape(self, product, units=1):
        self.escapes_by_product[product] = self.escapes_by_product.get(product, 0) + units
    def record_shipment(self, product, qty_good_est, qty_suspect_est):
        self.shipped_good[product] = self.shipped_good.get(product, 0) + int(qty_good_est)
        self.shipped_suspect[product] = self.shipped_suspect.get(product, 0) + int(qty_suspect_est)
    def schedule_potential_returns(self, order_id, product, qty_suspect):
        if qty_suspect <= 0 or not self.cfg.get("returns", {}).get("enabled", True):
            return
        import random
        delay_h = float(self.cfg["returns"].get("delay_h", 24.0))
        yield self.env.timeout(delay_h)
        p_return = float(self.cfg["returns"].get("escape_to_return_p", 0.5))
        returns = 0
        for _ in range(qty_suspect):
            if random.random() < p_return:
                returns += 1
        if returns <= 0:
            return
        self.returns_by_product[product] = self.returns_by_product.get(product, 0) + returns
        self.return_events.append({
            "time": self.env.now, "order_id": order_id, "product": product, "qty": returns
        })
        if self.financial:
            refund_fraction = float(self.cfg["returns"].get("refund_fraction", 1.0))
            self.financial.total_revenue -= refund_fraction * returns * PRODUCTS[product]["profit_per_unit"]
    def kpis(self):
        per_product = {}
        for product, c in self.product_counts.items():
            produced = max(1, c.get("produced", 0))  
            defects = c.get("defects_detected", 0)
            rework = c.get("rework", 0)
            scrap = c.get("scrap", 0)
            escapes = c.get("escapes", 0)
            shipped_good = self.shipped_good.get(product, 0)
            shipped_suspect = self.shipped_suspect.get(product, 0)
            returns = self.returns_by_product.get(product, 0)
            defect_rate = defects / produced
            rework_rate = rework / produced
            scrap_rate = scrap / produced
            escape_rate = escapes / produced
            fpy = (produced - defects - scrap) / produced
            rty = fpy
            return_rate = (returns / max(1, shipped_good + shipped_suspect))
            sat = 100.0
            sat -= 40.0 * return_rate
            sat -= 20.0 * escape_rate
            sat = max(0.0, min(100.0, sat))
            per_product[product] = {
                "produced": c.get("produced", 0),
                "defect_rate": defect_rate,
                "rework_rate": rework_rate,
                "scrap_rate": scrap_rate,
                "escape_rate": escape_rate,
                "fpy": fpy,
                "rty_proxy": rty,
                "shipped_good": shipped_good,
                "shipped_suspect": shipped_suspect,
                "returns": returns,
                "return_rate": return_rate,
                "customer_satisfaction": sat
            }
        return {"by_product": per_product}
def _validate_config(self):
    required_configs = ['INVENTORY_POLICY', 'RAW_MATERIALS', 'MATERIAL_CONSUMPTION', 'SHIPPING_CONFIG']
    for config_name in required_configs:
        if config_name not in globals():
            print(f"WARNING: {config_name} not found in configuration")
    for material, config in RAW_MATERIALS.items():
        required_keys = ['cost_per_unit', 'suppliers', 'lead_time']
        for key in required_keys:
            if key not in config:
                print(f"WARNING: {material} missing required key: {key}")
                if key == 'cost_per_unit':
                    config[key] = 1.0
                elif key == 'suppliers':
                    config[key] = ['DefaultSupplier']
                elif key == 'lead_time':
                    config[key] = (1, 3)
class InventoryManager:
    def __init__(self, env,cfg, financial_tracker, RAW_MATERIALS, INVENTORY_POLICY, environmental=None, job_logger=None, planning_data=None):
        self.env = env
        self.cfg = cfg
        self.financial_tracker = financial_tracker
        self.environmental = environmental
        self.job_logger = job_logger
        self.raw_materials = RAW_MATERIALS
        self.policy_config = INVENTORY_POLICY
        self.planning_data = planning_data
        self.stock = {}
        self.stock_movements = []
        self.procurement_orders = []
        self.stockouts = []
        self.finished_goods = {}
        self.shipped = {}
        self.inventory_value_log = []
        for material, config in RAW_MATERIALS.items():
            initial_stock = config.get('initial_stock', 0)
            self.stock[material] = initial_stock
        self.reorder_process = env.process(self._monitor_stock_levels())
        
    def _calculate_material_forecasts(self):
        if not self.planning_data or not self.planning_data.forecasted_demand_history: return {}
        product_forecasts, forecast_days = {}, len(self.planning_data.forecasted_demand_history)
        demand_cfg = self.env.cfg.PLANNING_CONFIG['DEMAND_MODEL']
        total_units = sum(self.planning_data.forecasted_demand_history)
        for p, r in demand_cfg['product_mix'].items(): product_forecasts[p] = total_units * r
        material_forecasts = {}
        for p_name, t_demand in product_forecasts.items():
            if p_name in self.env.cfg.MATERIAL_CONSUMPTION:
                for m_name, qpu in self.env.cfg.MATERIAL_CONSUMPTION[p_name].items():
                    material_forecasts.setdefault(m_name, 0); material_forecasts[m_name] += t_demand * qpu
        annualized = {}
        if forecast_days > 0:
            for m, d in material_forecasts.items(): annualized[m] = (d / forecast_days) * 365
        return annualized

    def _log_inventory_value(self):
        current_total_value = 0
        for material, level in self.stock.items():
            cost_per_unit = self.raw_materials.get(material, {}).get('cost_per_unit', 0)
            current_total_value += level * cost_per_unit
        self.inventory_value_log.append((self.env.now, current_total_value))
        
    def consume_materials(self, required_materials: dict, quantity: int = 1, product_type: str = "Unknown", stage: str = "Unknown"):
        shortages = []
        for mat, amt in required_materials.items():
            total_needed = amt * quantity
            if mat not in self.stock:
                print(f"ERROR: Material {mat} not in stock inventory. Initializing to 0.")
                self.stock[mat] = 0
            if self.stock[mat] < total_needed:
                shortage = total_needed - self.stock[mat]
                shortages.append((mat, shortage, total_needed))
        if shortages:
            for mat, shortage, needed in shortages:
                self.stockouts.append({
                    'time': self.env.now, 'material': mat,
                    'shortage': shortage, 'needed': needed,
                    'available': self.stock[mat],
                    'product': product_type, 'stage': stage
                })
                if self.financial_tracker:
                    self.financial_tracker.record_stockout_cost(mat, shortage)
                print(f"STOCKOUT: Not enough {mat} for {product_type} at {stage} - Need: {needed:.2f}, Have: {self.stock[mat]:.2f}")
            return False 
        for mat, amt in required_materials.items():
            consumed = amt * quantity
            self.stock[mat] -= consumed
            self.stock_movements.append({
                'time': self.env.now, 'material': mat, 'type': 'consumption',
                'quantity': -consumed, 'remaining_stock': self.stock[mat],
                'product': product_type, 'stage': stage
            })
            print(f"INVENTORY: Consumed {consumed:.2f} {mat} for {product_type} at {stage} - Stock: {self.stock[mat]:.2f}")

        self._log_inventory_value()
        return True 
        
    def get_stock_level(self, material):
        return self.stock.get(material, 0)
    def get_on_order_quantity(self, material):
        pending_orders = [
            po['quantity'] for po in self.procurement_orders 
            if po['material'] == material and po['status'] == 'pending'
        ]
        return sum(pending_orders)
    def add_procurement_order(self, order_details):
        self.procurement_orders.append(order_details)
    def receive_delivery(self, order_id):
        for order in self.procurement_orders:
            if order['order_id'] == order_id and order['status'] == 'pending':
                self.stock[order['material']] += order['quantity']
                order['status'] = 'delivered'
                order['actual_delivery'] = self.env.now

                self._log_inventory_value()
                
                print(f"DELIVERY: Received {order['quantity']:.2f} {order['material']} - Stock: {self.stock[order['material']]:.2f}")
                return order
        return None
    def receive_finished_unit(self, product_type, quantity=1):
        self.finished_goods[product_type] = self.finished_goods.get(product_type, 0) + quantity
        print(f"FG: Received {quantity} unit(s) of {product_type}. FG total: {self.finished_goods[product_type]}")
    def ship_available_for_order(self, order):
        need = {}
        for it in order['items']:
            if it['status'] == 'completed':
                need[it['product']] = need.get(it['product'], 0) + 1
        shipped_count = 0
        if not SHIPPING_CONFIG.get('ship_partial_orders', True):
            for p, q in need.items():
                if self.finished_goods.get(p, 0) < q:
                    return 0
        for p, q in need.items():
            available = self.finished_goods.get(p, 0)
            to_ship = min(q, available)
            if to_ship > 0:
                self.finished_goods[p] = available - to_ship
                self.shipped[p] = self.shipped.get(p, 0) + to_ship
                shipped_count += to_ship
        return shipped_count
    def get_inventory_status(self):
        status = {}
        for mat, lvl in self.stock.items():
            policy = self.policy_config.get(mat, {})
            policy_type = policy.get('policy')
            rop = 0
            target = 1
            if policy_type == 's_S':
                rop = policy.get('s', 0)
                target = policy.get('S', 1)
            elif policy_type == 'BASE_STOCK':
                target = policy.get('target_level', 1)
                rop = target * 0.25  
            elif policy_type == 'REORDER_POINT':
                rop = policy.get('reorder_level', 0)
                target = rop + policy.get('order_qty', 1)
            else:
                target = max(1, lvl) 
            status[mat] = {
                'current_stock': lvl,
                'reorder_point': rop,
                'target_level': target,
                'stock_ratio': (lvl / target) if target > 0 else 0,
                'status': 'Low' if lvl <= rop else 'Normal'
            }
        return status
    def get_inventory_metrics(self):
        total_proc_cost = sum(o['cost'] for o in self.procurement_orders)
        total_stockouts = len(self.stockouts)
        avg_stock_levels = {
            m: (sum(ev['remaining_stock'] for ev in self.stock_movements if ev['material'] == m) /
                max(1, len([ev for ev in self.stock_movements if ev['material'] == m])))
            for m in self.raw_materials.keys()
        }
        metrics = {
            'total_procurement_cost': total_proc_cost,
            'total_stockouts': total_stockouts,
            'total_orders_placed': len(self.procurement_orders),
            'avg_stock_levels': avg_stock_levels
        }
        return metrics
    def get_total_stock_level(self):
        return sum(self.stock.values())
    def _monitor_stock_levels(self):
        print("\n--- Starting Hybrid Inventory Monitoring Process ---")
        self.effective_policies = {}
        material_forecasts = self._calculate_material_forecasts()
        for name, details in self.raw_materials.items():
            policy_cfg = self.policy_config.get(name, self.policy_config.get('default', {}))
            method = policy_cfg.get('policy') 
            final_policy = {'policy': method}
            if method == 'EOQ' and material_forecasts and name in material_forecasts:
                print(f"  - {name}: Calculating dynamic 'EOQ' policy.")
                annual_demand = material_forecasts[name]
                eoq_qty = calculate_eoq(annual_demand, details['ordering_cost'], details['holding_cost'])
                avg_daily_demand = annual_demand / 365
                demand_during_lead_time = avg_daily_demand * details['lead_time_days']
                safety_stock = avg_daily_demand * policy_cfg.get('params_eoq', {}).get('safety_stock_days', 0)
                final_policy.update({'policy': 'REORDER_POINT', 'reorder_level': round(demand_during_lead_time + safety_stock), 'order_qty': round(eoq_qty)})
            else:
                if method == 'EOQ': print(f"  - {name}: WARNING - 'EOQ' for '{name}' falling back to fixed params.")
                else: print(f"  - {name}: Using preserved '{method}' policy.")
                final_policy.update(policy_cfg)
            self.effective_policies[name] = final_policy
        print("------------------------------------")
        while True:
            for item_name in self.stock.keys():
                policy = self.effective_policies[item_name]
                method = policy.get('policy')
                level = self.get_stock_level(item_name)
                on_order = self.get_on_order_quantity(item_name)
                if on_order > 0:
                    continue
                order_qty = 0
                if method == 's_S' and level <= policy['s']:
                    order_qty = policy['S'] - level
                elif method == 'REORDER_POINT' and level <= policy['reorder_level']:
                    order_qty = policy['order_qty']
                elif method == 'BASE_STOCK':
                    if self.env.now > 0 and self.env.now % policy['review_period_h'] == 0:
                        order_qty = policy['target_level'] - level
                if order_qty > 0:
                    capacity = self.raw_materials[item_name].get('storage_capacity', self.policy_config.get('storage_capacity', float('inf')))
                    if (level + order_qty) > capacity:
                        print(f"INVENTORY ALERT ({method}): Order for {item_name} ({order_qty} units) BLOCKED. Would exceed capacity {capacity}.")
                    else:
                        print(f"INVENTORY ({method}): {item_name} stock ({level}) triggered order for {order_qty} units.")
                        self.env.process(self.replenish(item_name, order_qty))
            
            yield self.env.timeout(1) 
    def _get_lead_time_from_supplier(self, supplier_name):
        try:
            supplier_data = self.cfg.SUPPLY_CHAIN_CONFIG['suppliers'][supplier_name]
            dist_cfg = supplier_data['lead_time_dist']
            dist_type = dist_cfg['type']
            if dist_type == 'normal':
                return max(1, random.normalvariate(dist_cfg['mean'], dist_cfg['stddev']))
            elif dist_type == 'triangular':
                return max(1, random.triangular(dist_cfg['low'], dist_cfg['high'], dist_cfg['mode']))
            elif dist_type == 'lognormal':
                return max(1, random.lognormvariate(dist_cfg['mean'], dist_cfg['stddev']))
            else: 
                return dist_cfg.get('value', 24 * 7) 
        except (KeyError, AttributeError):
            print(f"WARNING: Could not find lead time distribution for supplier '{supplier_name}'. Defaulting to 7 days.")
            return 24 * 7
    def replenish(self, material, quantity):
        policy_cfg = self.policy_config.get(material, {})
        available_suppliers = policy_cfg.get('available_suppliers')
        if not available_suppliers:
            print(f"FATAL ERROR: No suppliers defined for material '{material}'. Cannot replenish.")
            return 
        procurement_policy = self.cfg.SUPPLY_CHAIN_CONFIG.get('procurement_policy', 'lowest_cost')
        best_supplier_name = None
        if procurement_policy == 'best_reliability':
            best_supplier_name = max(available_suppliers, key=lambda s: self.cfg.SUPPLY_CHAIN_CONFIG['suppliers'][s]['reliability'])
        else: 
            best_supplier_name = min(available_suppliers, key=lambda s: self.cfg.SUPPLY_CHAIN_CONFIG['suppliers'][s]['cost_multiplier'])
        item_details = self.raw_materials[material]
        supplier_details = self.cfg.SUPPLY_CHAIN_CONFIG['suppliers'][best_supplier_name]
        order_id = f"PO-{len(self.procurement_orders) + 1}-{material}"
        cost = (item_details['cost_per_unit'] * supplier_details['cost_multiplier']) * quantity
        order_details = {
            'order_id': order_id, 'material': material, 'quantity': quantity,
            'status': 'pending', 'order_time': self.env.now, 'cost': cost,
            'supplier': best_supplier_name
        }
        self.add_procurement_order(order_details)
        lead_time_hours = self._get_lead_time_from_supplier(best_supplier_name)
        print(f"INVENTORY: Placed order {order_id} for {quantity} {material} from {best_supplier_name}. Estimated lead time: {lead_time_hours:.2f} hours.")
        yield self.env.timeout(lead_time_hours)
        self.receive_delivery(order_id)
class SupplyChainManager:
    def __init__(self, env, inventory_manager, financial_tracker, supply_chain_cfg, inventory_policy_cfg,storage_capacity=None):
        self.env = env
        self.inventory_manager = inventory_manager
        self.financial_tracker = financial_tracker
        self.supply_cfg = supply_chain_cfg
        self.inventory_policies = inventory_policy_cfg
        self.storage_capacity= storage_capacity
        self.suppliers = self.supply_cfg['suppliers']
        self.active_disruptions = {}
        self.order_counter = 0
        if self.supply_cfg.get('risk_model') == 'dynamic_events':
            self.env.process(self.world_events_monitor())
    def world_events_monitor(self):
        for event in self.supply_cfg.get('disruption_events', []):
            if self.env.now < event['start_time']:
                yield self.env.timeout(event['start_time'] - self.env.now)
            print(f"!!! WORLD EVENT START (t={self.env.now:.2f}): {event['name']} impacting region {event['target_region']}")
            self.active_disruptions[event['target_region']] = event['impact']
            yield self.env.timeout(event['duration'])
            print(f"--- WORLD EVENT END (t={self.env.now:.2f}): {event['name']} no longer impacting {event['target_region']}")
            if event['target_region'] in self.active_disruptions:
                del self.active_disruptions[event['target_region']]
    def get_supplier_performance(self, supplier_id):
        perf_config = {k: v.copy() if isinstance(v, dict) else v for k, v in self.suppliers[supplier_id].items()}
        region = perf_config['region']
        if self.supply_cfg['risk_model'] == 'dynamic_events' and region in self.active_disruptions:
            impact = self.active_disruptions[region]
            perf_config['reliability'] *= impact.get('reliability_multiplier', 1.0)
            lt_config = perf_config['lead_time']
            lt_config['mean'] *= impact.get('lead_time_multiplier', 1.0)
            lt_config['stddev'] *= impact.get('lead_time_multiplier', 1.0)
        return perf_config
    def choose_supplier(self, material):
        available_ids = self.inventory_policies[material]['available_suppliers']
        policy = self.supply_cfg.get('procurement_policy', 'best_reliability') 
        best_supplier_id = None
        if not available_ids:
            print(f"!!! PROCUREMENT ERROR: No available suppliers listed for {material}!")
            return None, None
        if policy == 'best_reliability':
            best_supplier_id = max(
                available_ids, 
                key=lambda s_id: self.get_supplier_performance(s_id)['reliability']
            )
        elif policy == 'lowest_cost':
            best_supplier_id = min(
                available_ids, 
                key=lambda s_id: self.get_supplier_performance(s_id)['cost_multiplier']
            )
        else:
            print(f"WARNING: Invalid procurement_policy '{policy}'. Defaulting to first available.")
            best_supplier_id = available_ids[0]
        return best_supplier_id, self.get_supplier_performance(best_supplier_id)
    def _draw_lead_time(self, supplier_performance_config):
        dist_config = supplier_performance_config.get('lead_time_dist')
        if not dist_config or 'type' not in dist_config:
            return random.uniform(24, 48)
        dist_type = dist_config['type']
        try:
            if dist_type == 'normal':
                return random.normalvariate(dist_config['mean'], dist_config['stddev'])
            elif dist_type == 'triangular':
                return random.triangular(dist_config['low'], dist_config['high'], dist_config['mode'])
            elif dist_type == 'lognormal':
                return random.lognormvariate(dist_config['mean'], dist_config['stddev'])
            elif dist_type == 'uniform':
                return random.uniform(dist_config['low'], dist_config['high'])
            else:
                print(f"WARNING: Unknown lead time distribution type '{dist_type}'. Using default.")
                return random.uniform(24, 48)
        except KeyError as e:
            print(f"ERROR: Missing parameter {e} for '{dist_type}' lead time distribution. Using default.")
            return random.uniform(24, 48)
    def _trigger_procurement(self, material, quantity):
        if quantity <= 0:
            return
        if self.storage_capacity is not None:
            current_total_stock = self.inventory_manager.get_total_stock_level()
            available_space = self.storage_capacity - current_total_stock
            if available_space <= 0:
                print(f"!!! PROCUREMENT BLOCKED (t={self.env.now:.2f}): No warehouse space available for {material}.")
                return 
            if quantity > available_space:
                print(f"INFO (t={self.env.now:.2f}): Order for {material} reduced from {quantity:.2f} to {available_space:.2f} due to storage capacity limits.")
                quantity = available_space 
        self.order_counter += 1
        supplier_id, perf = self.choose_supplier(material)
        if not supplier_id:
            return
        moq = perf.get('min_order_qty', 0)  
        final_order_quantity = max(quantity, moq)
        if final_order_quantity > quantity:
            print(f"INFO (t={self.env.now:.2f}): Order for {material} rounded up from {quantity:.2f} to {final_order_quantity:.2f} to meet MOQ of {supplier_id}.")
        quantity_to_order = final_order_quantity
        base_cost = self.inventory_manager.raw_materials[material]['cost_per_unit']
        cost = float(quantity_to_order) * base_cost * perf['cost_multiplier']
        lead_time = self._draw_lead_time(perf)
        lead_time = max(1, lead_time)
        order = {
            'order_id': f"PO_{self.order_counter}", 'material': material, 'quantity': float(quantity_to_order),
            'supplier': supplier_id, 'order_time': self.env.now, 'expected_delivery': self.env.now + lead_time,
            'cost': cost, 'status': 'pending'
        }
        self.inventory_manager.add_procurement_order(order)
        self.financial_tracker.record_procurement_cost(material, cost, float(quantity_to_order))
        print(f"--> PROCUREMENT (t={self.env.now:.2f}): Ordering {quantity_to_order:.2f} of {material} from {supplier_id} (ETA: {lead_time:.2f}h)")
        self.env.process(self._simulate_delivery(order, perf['reliability']))
    def _simulate_delivery(self, order, reliability):
        yield self.env.timeout(order['expected_delivery'] - self.env.now)
        if random.random() < reliability:
            self.inventory_manager.receive_delivery(order['order_id'])
        else:
            order['status'] = 'failed'
            print(f"!!! DISRUPTION (t={self.env.now:.2f}): Order {order['order_id']} for {order['material']} was lost!")
    def inventory_monitoring_process(self):
        for material, policy in self.inventory_policies.items():
            if not isinstance(policy, dict):
                continue  
            if policy.get('policy') == 'BASE_STOCK':
                self.env.process(self._periodic_review_process(material, policy))
        while True:
            yield self.env.timeout(1) 
            for material, policy in self.inventory_policies.items():
                if not isinstance(policy, dict):
                    continue
                policy_type = policy.get('policy')
                if policy_type not in ['s_S', 'REORDER_POINT']:
                    continue
                inventory_position = self.inventory_manager.get_stock_level(material) + self.inventory_manager.get_on_order_quantity(material)
                if policy_type == 's_S':
                    if inventory_position <= policy['s']:
                        order_qty = policy['S'] - inventory_position
                        self._trigger_procurement(material, order_qty)
                elif policy_type == 'REORDER_POINT':
                    if inventory_position <= policy['reorder_level']:
                        order_qty = policy['order_qty']
                        self._trigger_procurement(material, order_qty)
    def _periodic_review_process(self, material, policy_cfg):
        review_period = policy_cfg.get('review_period_h', 24)
        target_level = policy_cfg.get('target_level', 0)
        while True:
            yield self.env.timeout(review_period)
            inventory_position = self.inventory_manager.get_stock_level(material) + self.inventory_manager.get_on_order_quantity(material)
            if inventory_position < target_level:
                order_qty = target_level - inventory_position
                self._trigger_procurement(material, order_qty)

                
class TransportationManager:
    def __init__(self, env, worker_pool, financial_tracker, inventory_manager, config,environmental=None):
        self.env = env
        self.workerpool = worker_pool
        self.financial = financial_tracker
        self.inventory = inventory_manager
        self.cfg = config
        self.environmental = environmental
        self.graph = {}
        for e in self.cfg["floor_edges"]:
            u, v = e["u"], e["v"]
            if "conveyor" in e.get("allowed_modes", []):
                continue
            self.graph.setdefault(u, []).append((v, e))
            self.graph.setdefault(v, []).append((u, e))
        self.inbound_buffers = {}
        for stage, meta in self.cfg["stage_buffers"].items():
            cap = int(meta.get("inbound_capacity", 100))
            self.inbound_buffers[stage] = simpy.Store(self.env, capacity=cap)
        self.modes = self.cfg["modes"]
        self.vehicle_resources = {m: simpy.Resource(self.env, capacity=mcfg["fleet_size"])
                                  for m, mcfg in self.modes.items()}
        self.conveyors = []
        for seg in self.cfg.get("conveyors", []):
            q = simpy.Store(self.env, capacity=seg["max_wip_units"])
            self.conveyors.append({"seg": seg, "q": q})
            self.env.process(self._run_conveyor(seg, q))
        self._sp_cache = {}
    def request_transport(self, stage, qty_units=1, mode_hint=None):
        evt = self.env.event()
        self.env.process(self._dispatch(stage, qty_units, mode_hint, evt))
        return evt
    def _node_for_stage(self, stage):
        return self.cfg["stage_buffers"][stage]["node"]
    def _shortest_time_path(self, u, v, mode):
        key = (u, v, mode)
        if key in self._sp_cache:
            return self._sp_cache[key]
        import heapq
        speed = self.modes[mode]["speed_mps"]
        pq = [(0.0, u, [])]
        seen = set()
        best = None
        while pq:
            t, cur, path = heapq.heappop(pq)
            if (cur, mode) in seen:
                continue
            seen.add((cur, mode))
            if cur == v:
                best = (t, path)
                break
            for nxt, e in self.graph.get(cur, []):
                if mode not in e.get("allowed_modes", []):
                    continue
                length = float(e["length_m"])
                dt_h = (length / speed) / 3600.0
                heapq.heappush(pq, (t + dt_h, nxt, path + [(cur, nxt, e)]))
        if best is None:
            self._sp_cache[key] = (float("inf"), [])
        else:
            self._sp_cache[key] = best
        return self._sp_cache[key]
    def _dispatch(self, stage, qty_units, mode_hint, done_evt):
        wh = self.cfg["warehouse_node"]
        dst = self._node_for_stage(stage)
        mode = mode_hint or "forklift"
        eta_h, path = self._shortest_time_path(wh, dst, mode)
        if not path or eta_h == float("inf"):
            for alt in self.modes.keys():
                if alt == mode:
                    continue
                eta2, path2 = self._shortest_time_path(wh, dst, alt)
                if path2 and eta2 < float("inf"):
                    mode, eta_h, path = alt, eta2, path2
                    break
        mcfg = self.modes[mode]
        remaining = qty_units
        arrived = 0
        while remaining > 0:
            load_units = min(remaining, mcfg["capacity_units"])
            remaining -= load_units
            with self.vehicle_resources[mode].request() as req:
                yield req
                yield from self._do_load_unload(hours=mcfg["load_time_h"])
                total_m = sum(e["length_m"] for _, _, e in path)
                travel_h = eta_h
                yield self.env.timeout(travel_h * 3600.0)
                yield from self._do_load_unload(hours=mcfg["unload_time_h"])
                for _ in range(load_units):
                    yield self.inbound_buffers[stage].put({
                        "stage": stage, "from": "WH", "mode": mode, "arrived": self.env.now
                    })
                if self.financial:
                    self.financial.record_transport_cost(
                        self.env,
                        mode=mode,
                        time_hours=mcfg["load_time_h"] + travel_h + mcfg["unload_time_h"],
                        distance_km=total_m / 1000.0,
                        operating_cost_per_hour=mcfg["operating_cost_per_hour"],
                        energy_cost_per_km=mcfg["energy_cost_per_km"]
                    )
                if hasattr(self, "environmental"):
                    self.environmental.record_transport(mode, distance_km, time_hours)
                arrived += load_units
        if not done_evt.triggered:
            done_evt.succeed(arrived)
    def _do_load_unload(self, hours):
        def _inner():
            if self.workerpool is not None and hasattr(self.workerpool, "request_worker"):
                req = self.workerpool.request_worker("TransportDock")
                yield req  
                try:
                    yield self.env.timeout(hours * 3600.0)
                finally:
                    if hasattr(self.workerpool, "release_worker"):
                        if hasattr(req, "workerid"):
                            self.workerpool.release_worker(req.workerid, machine="TransportDock")
                        else:
                            self.workerpool.release_worker("TransportDock")
            else:
                yield self.env.timeout(hours * 3600.0)
        return _inner()
    def _run_conveyor(self, seg, q):
        while True:
            token = yield q.get()
            travel_s = seg["length_m"] / seg["speed_mps"]
            yield self.env.timeout(travel_s)
            dst_node = seg["v"]
            stage = None
            for stg, meta in self.cfg["stage_buffers"].items():
                if meta["node"] == dst_node:
                    stage = stg
                    break
            if stage is None:
                continue
            yield self.inbound_buffers[stage].put(token)
    def push_conveyor(self, u, v, token):
        for c in self.conveyors:
            s = c["seg"]
            if s["u"] == u and s["v"] == v:
                return c["q"].put(token)
        raise RuntimeError(f"No conveyor segment {u}->{v}")
def _sample_from_dist(dist_config):
    dist_type = dist_config.get('type', 'uniform')
    if dist_type == 'discrete':
        values = dist_config['values']
        probs = dist_config['probs']
        r = random.random()
        cumulative = 0.0
        for val, prob in zip(values, probs):
            cumulative += prob
            if r <= cumulative:
                return val
        return values[-1]
    elif dist_type == 'uniform':
        return random.uniform(dist_config['low'], dist_config['high'])
    else:
        return dist_config.get('default', 1.0)
class Worker:
    def __init__(self, env, worker_id, name, skills, experience_years, base_efficiency, preferred_shift=0, calendar=None, worker_pool_ref=None, metrics_ref=None):
        self.env = env
        self.worker_id = worker_id
        self.name = name
        self.skills = skills
        self.experience_years = experience_years
        self.base_efficiency = base_efficiency
        self.preferred_shift = preferred_shift
        self.calendar = calendar
        self.worker_pool_ref = worker_pool_ref
        self.metrics_ref = metrics_ref
        self.is_available = True
        self.is_working = False
        self.current_machine = None
        self.current_task_start = None
        self.fatigue_level = 0.0
        self.consecutive_work_hours = 0.0
        self.is_sick = False
        self.is_on_vacation = False
        self.sick_until = None
        self.vacation_until = None
        self.late_arrival_delay = 0.0
        self.work_sessions = []
        self.total_work_hours = 0.0
        self.tasks_completed = 0
        self.efficiency_history = []
        self.timeline = StateTimeline(f"Worker_{self.name}")
        self.timeline.set_state(0.0, 'available')
        self.fatigue_process = env.process(self._manage_fatigue())
        self.unpredictability_process = env.process(self._manage_unpredictability())
    def can_operate_machine(self, machine_type):
        for skill in self.skills:
            if skill in WORKER_SKILLS and machine_type in WORKER_SKILLS[skill]['compatible_machines']:
                return True
        return False
    def get_efficiency_for_machine(self, machine_type):
        if not self.can_operate_machine(machine_type):
            return 0.0
        best_efficiency = 0.0
        for skill in self.skills:
            if skill in WORKER_SKILLS:
                skill_config = WORKER_SKILLS[skill]
                if machine_type in skill_config['compatible_machines']:
                    skill_efficiency = skill_config['efficiency_multiplier']
                    best_efficiency = max(best_efficiency, skill_efficiency)
        fatigue_penalty = self.fatigue_level * 0.3
        total_efficiency = self.base_efficiency * best_efficiency * (1.0 - fatigue_penalty)
        return max(0.1, total_efficiency)
    def is_currently_available(self):
            if not self.is_available:
                return False
            if self.is_working:
                return False 
            if self.is_sick or self.is_on_vacation:
                return False
            if self.fatigue_level > WORKER_POOL_CONFIG['max_fatigue_threshold']:
                return False
            return True
    def _is_in_working_hours(self):
        if not self.calendar:
            return True
        effective_now = self.env.now - self.late_arrival_delay
        return self.calendar.is_working_time(effective_now)
    def assign_to_machine(self, machine, task_type='operation'):
        if not self.is_currently_available():
            return False
        self.is_working = True
        self.current_machine = machine
        self.current_task_start = self.env.now
        self.timeline.set_state(self.env.now, f'working_{task_type}')
        try:
            if self.worker_pool_ref and self.metrics_ref:
                total = len(self.worker_pool_ref.workers)
                if total > 0:
                    working = sum(1 for w in self.worker_pool_ref.workers.values() if w.is_working)
                    self.metrics_ref.observe("worker_utilization", working / total, self.env.now)
        except Exception as e:
            print(f"DEBUG: An error occurred in assign_to_machine metric recording: {e}")
        return True
    def release_from_machine(self):
        if self.is_working and self.current_task_start is not None:
            try:
                if self.worker_pool_ref and self.metrics_ref:
                    total = len(self.worker_pool_ref.workers)
                    if total > 0:
                        working = sum(1 for w in self.worker_pool_ref.workers.values() if w.is_working)
                        self.metrics_ref.observe("worker_utilization", working / total, self.env.now)
            except Exception as e:
                print(f"DEBUG: An error occurred in release_from_machine metric recording: {e}")
            work_duration = self.env.now - self.current_task_start
            self.consecutive_work_hours += work_duration
            self.total_work_hours += work_duration
            self.tasks_completed += 1
            self.work_sessions.append({
                'start': self.current_task_start, 'end': self.env.now, 'duration': work_duration,
                'machine': self.current_machine.name if self.current_machine else 'Unknown',
                'fatigue_at_start': self.fatigue_level
            })
            self.is_working = False
            self.current_machine = None
            self.current_task_start = None
            self.timeline.set_state(self.env.now, 'available')
    def _manage_fatigue(self):
        while True:
            yield self.env.timeout(0.25)
            if self.is_working:
                fatigue_increase = 0.05
                self.fatigue_level = min(1.0, self.fatigue_level + fatigue_increase)
            else:
                if WORKER_POOL_CONFIG['fatigue_enabled']:
                    recovery_rate = WORKER_POOL_CONFIG['fatigue_recovery_rate']
                    self.fatigue_level = max(0.0, self.fatigue_level - recovery_rate * 0.25)
    def _manage_unpredictability(self):
        while True:
            yield self.env.timeout(24)
            if not WORKER_POOL_CONFIG['unpredictability_enabled']:
                continue
            current_time = self.env.now
            if self.is_sick and self.sick_until and current_time >= self.sick_until:
                self.is_sick = False
                self.sick_until = None
                self.timeline.set_state(current_time, 'available')
            if self.is_on_vacation and self.vacation_until and current_time >= self.vacation_until:
                self.is_on_vacation = False
                self.vacation_until = None
                self.timeline.set_state(current_time, 'available')
            if not (self.is_sick or self.is_on_vacation):
                unpredict_config = WORKER_UNPREDICTABILITY
                if random.random() < unpredict_config['sick_day_probability']:
                    duration = _sample_from_dist(unpredict_config['sick_duration_dist'])
                    self.is_sick = True
                    self.sick_until = current_time + duration * 24
                    self.timeline.set_state(current_time, 'sick')
                elif random.random() < unpredict_config['vacation_probability']:
                    duration = _sample_from_dist(unpredict_config['vacation_duration_dist'])
                    self.is_on_vacation = True
                    self.vacation_until = current_time + duration * 24
                    self.timeline.set_state(current_time, 'vacation')
class WorkerPool:
    def __init__(self, env, calendar, financial_tracker, worker_definitions_cfg, rt_metrics_bus):
        self.env = env
        self.calendar = calendar
        self.financial_tracker = financial_tracker
        self.worker_definitions = worker_definitions_cfg
        self.rt_metrics_bus = rt_metrics_bus
        self.workers = {}
        self.skill_matrix = {}
        self.assignment_history = []
        self.waiting_requests = []
        self.in_use = {}
        self._initialize_workers()
        self._build_skill_matrix()
        self.scheduling_process = env.process(self._continuous_scheduling())
    def _initialize_workers(self):
        for worker_info in self.worker_definitions:
            worker_id = worker_info['id']
            worker = Worker(
                env=self.env,
                worker_id=worker_id,
                name=worker_info.get('name', worker_id),
                skills=worker_info.get('skills', []),
                experience_years=worker_info.get('experience_years', 1),
                base_efficiency=worker_info.get('base_efficiency', 1.0),
                preferred_shift=worker_info.get('preferred_shift', 0),
                calendar=self.calendar,
                worker_pool_ref=self,
                metrics_ref=self.rt_metrics_bus
            )
            self.workers[worker_id] = worker
            for skill in worker.skills:
                if skill not in self.skill_matrix:
                    self.skill_matrix[skill] = []
                self.skill_matrix[skill].append(worker)
    def _build_skill_matrix(self):
        self.skill_matrix = {}
        for machine_type in MACHINE_DEFINITIONS.keys():
            self.skill_matrix[machine_type] = []
            for worker in self.workers.values():
                if worker.can_operate_machine(machine_type):
                    self.skill_matrix[machine_type].append(worker)
    def request_worker(self, machine_type, priority='normal'):
        request = {
            'machine_type': machine_type,
            'priority': priority,
            'request_time': self.env.now,
            'assigned_worker': None,
            'event': self.env.event()
        }
        self.waiting_requests.append(request)
        return request['event']
    def release_worker(self, worker_id, machine):
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.release_from_machine()
    def _continuous_scheduling(self):
        while True:
            if self.waiting_requests:
                fulfilled_requests = []
                for request in self.waiting_requests:
                    worker = self._find_best_worker(request['machine_type'])
                    if worker:
                        request['assigned_worker'] = worker
                        request['event'].succeed(worker)
                        fulfilled_requests.append(request)
                        self.assignment_history.append({
                            'time': self.env.now,
                            'worker_id': worker.worker_id,
                            'machine_type': request['machine_type'],
                            'wait_time': self.env.now - request['request_time'],
                            'algorithm_used': WORKER_SCHEDULING['algorithm']
                        })
                    else:
                        print(f"DEBUG (t={self.env.now:.2f}): Scheduler could NOT FIND any available worker for stage: {request['machine_type']}")
                self.waiting_requests = [r for r in self.waiting_requests if r not in fulfilled_requests]
            yield self.env.timeout(0.1)
    def _find_best_worker(self, machine_type):
        available_workers = [w for w in self.skill_matrix.get(machine_type, []) 
                           if w.is_currently_available()]
        if not available_workers:
            return None
        algorithm = WORKER_SCHEDULING['algorithm']
        if algorithm == 'skill_priority':
            return self._skill_priority_selection(available_workers, machine_type)
        elif algorithm == 'efficiency_based':
            return self._efficiency_based_selection(available_workers, machine_type)
        else:
            return available_workers[0]
    def _skill_priority_selection(self, workers, machine_type):
        weights = WORKER_SCHEDULING['priority_weights']
        scored_workers = []
        for worker in workers:
            machine_reqs = MACHINE_WORKER_REQUIREMENTS.get(machine_type, {})
            preferred_skills = machine_reqs.get('preferred_skills', [])
            skill_score = sum(1.0 for skill in worker.skills if skill in preferred_skills)
            skill_score = min(1.0, skill_score / len(preferred_skills)) if preferred_skills else 0.5
            efficiency_score = worker.get_efficiency_for_machine(machine_type) / 1.5
            fatigue_score = 1.0 - worker.fatigue_level
            experience_score = min(1.0, worker.experience_years / 10.0)
            score = (weights['skill_match'] * skill_score +
                    weights['efficiency'] * efficiency_score +
                    weights['fatigue_level'] * fatigue_score +
                    weights['experience'] * experience_score)
            scored_workers.append((score, worker))
        scored_workers.sort(key=lambda x: x[0], reverse=True)
        return scored_workers[0][1]
    def _efficiency_based_selection(self, workers, machine_type):
        best_worker = None
        best_efficiency = 0.0
        for worker in workers:
            efficiency = worker.get_efficiency_for_machine(machine_type)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_worker = worker
        return best_worker
    def get_worker_utilization(self):
        total_workers = len(self.workers)
        working_workers = sum(1 for w in self.workers.values() if w.is_working)
        available_workers = sum(1 for w in self.workers.values() if w.is_currently_available())
        return {
            'total_workers': total_workers,
            'working': working_workers,
            'available': available_workers,
            'utilization_rate': working_workers / total_workers if total_workers > 0 else 0
        }
class PlanningData:
    def __init__(self):
        self.actual_demand_history = []
        self.forecasted_demand_history = []
class EnhancedMachine(simpy.Resource):
    def __init__(self, env, name, stage, capacity=1, mtbf=100, mttr=10, maintenance_interval=200, 
                 financial_tracker=None, metrics_bus=None, failure_model='exponential', weibull_k=None, 
                 weibull_lambda=None, lognorm_mu=None, lognorm_sigma=None,cfg=None, machines=None):
        self.env = env
        self.cfg = cfg
        self.machines = machines
        self.name = name
        self.stage = stage
        self._capacity = capacity
        self.financial_tracker = financial_tracker
        self.metrics_bus = metrics_bus
        self.last_product_type = None
        self.current_users = set()
        run_queuing_config = self.cfg.QUEUING_CONFIG
        policy_config = run_queuing_config.get('stage_policies', {}).get(self.stage, run_queuing_config['default_policy'])
        print(f"[DEBUG] Machine '{self.name}' at stage '{self.stage}' is using policy: {policy_config.get('policy')}")
        buffer_cap = policy_config.get('buffer_capacity', float('inf'))
        policy = get_policy_from_config(policy_config)
        self.queue = PolicyQueue(self.env, policy, self.name, self.metrics_bus, capacity=buffer_cap)
        self.resource = simpy.Resource(self.env, capacity=self._capacity)
        self.broken = False
        self.under_maintenance = False
        self.mtbf = mtbf
        self.mttr = mttr
        self.maintenance_interval = maintenance_interval
        self.total_downtime = 0
        self.breakdown_count = 0
        self.maintenance_count = 0
        self.downtime_events = []
        self.timeline = StateTimeline(self.name)
        self.units_produced_count = 0
        self.arrival_timestamps = []
        self.service_times = []
        self.analytical_results = {}
        self.analytical_params = {} 
        self.config = MACHINE_DEFINITIONS.get(self.stage, {})
        self.model_config = self.config.get("analytical_model", {})
        self.failure_model = failure_model
        self.weibull_k = weibull_k
        self.weibull_lambda = weibull_lambda
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma
        self.dispatcher_process = env.process(self._dispatcher())
        self.breakdown_process = env.process(self._breakdown_generator())
        self.maintenance_process = env.process(self._maintenance_scheduler())
    def _dispatcher(self):
        while True:
            job_to_process = yield self.env.process(self.queue.dequeue())
            routing = job_to_process.routing
            current_stage_index = routing.index(self.stage)
            is_last_stage = current_stage_index == len(routing) - 1
            if not is_last_stage:
                next_stage_name = routing[current_stage_index + 1]
                all_machines = job_to_process.context.get('machines', {})
                next_machine = pick_machine(all_machines[next_stage_name])
                next_queue = next_machine.queue
                while len(next_queue) >= next_queue.capacity:
                    print(f"!! BLOCKED: {self.name} waiting for space at {next_machine.name} (Buffer: {len(next_queue)}/{next_queue.capacity})")
                    self.timeline.set_state(self.env.now, 'blocked')
                    yield next_queue.get_event
            if job_to_process.request_event and not job_to_process.request_event.triggered:
                job_to_process.request_event.succeed({'status': 'ok'})
    def request(self, job: Job):
        job.request_event = self.env.event()
        self.queue.enqueue(job)
        return job.request_event
    def add_user(self, process):
        self.current_users.add(process)
    def remove_user(self, process):
        self.current_users.discard(process)
    def interrupt_current_users(self):
        for user_process in list(self.current_users):
            try:
                if user_process.is_alive and hasattr(user_process, 'interrupt'):
                    user_process.interrupt('machine_down')
            except RuntimeError:
                pass
    def record_unit_produced(self):
        self.units_produced_count += 1
    def draw_time_to_failure(self):
        if self.failure_model == 'weibull' and self.weibull_k and self.weibull_lambda:
            u = random.random()
            return self.weibull_lambda * ((-math.log(u)) ** (1.0 / self.weibull_k))
        elif self.failure_model == 'lognormal' and self.lognorm_mu and self.lognorm_sigma:
            return random.lognormvariate(self.lognorm_mu, self.lognorm_sigma)
        else:
            return random.expovariate(1.0 / self.mtbf)
    def _breakdown_generator(self):
        while True:
            time_to_failure = self.draw_time_to_failure()
            yield self.env.timeout(time_to_failure)
            if not self.broken and not self.under_maintenance:
                self.broken = True
                self.breakdown_count += 1
                breakdown_start = self.env.now
                self.timeline.set_state(self.env.now, 'breakdown')
                if self.financial_tracker:
                    self.financial_tracker.record_downtime_cost(self.name, 'breakdown', breakdown_start, self.mttr)
                self.downtime_events.append({
                    'type': 'breakdown',
                    'start_time': breakdown_start,  
                    'duration': self.mttr,
                    'cost': self.mttr * FINANCIAL_CONFIG.get('downtime_cost_per_hour', 40)
                })
                self.interrupt_current_users()
                print(f"BREAKDOWN: {self.name} failed at {self.env.now:.2f}")
                yield self.env.timeout(self.mttr)
                self.broken = False
                self.total_downtime += self.mttr
                self.timeline.set_state(self.env.now, 'idle')
                print(f"REPAIR COMPLETE: {self.name} restored at {self.env.now:.2f}")
    def _maintenance_scheduler(self):
        while True:
            yield self.env.timeout(self.maintenance_interval)
            if not self.broken:  
                self.under_maintenance = True
                self.maintenance_count += 1
                maint_start = self.env.now
                self.timeline.set_state(self.env.now, 'maintenance')
                maint_duration = self.mttr * 0.75
                stage_name = self.name.split('_')[0] if '_' in self.name else 'default'
                maint_cost = MACHINE_DEFINITIONS.get(stage_name, DEFAULT_MACHINE_CONFIG).get('maintenance_cost', 150)
                maint_duration = self.mttr * 0.75  
                if self.financial_tracker:
                    self.financial_tracker.record_downtime_cost(self.name, 'maintenance', maint_start, maint_duration)
                self.downtime_events.append({
                    'type': 'maintenance',
                    'start_time': maint_start,
                    'duration': maint_duration,
                    'cost': maint_cost
                })
                self.interrupt_current_users()
                print(f"MAINTENANCE: {self.name} starting maintenance at {self.env.now:.2f}")
                yield self.env.timeout(maint_duration)
                self.under_maintenance = False
                self.total_downtime += maint_duration
                self.timeline.set_state(self.env.now, 'idle')
                print(f"MAINTENANCE COMPLETE: {self.name} finished maintenance at {self.env.now:.2f}")
    def is_available(self):
        return not (self.broken or self.under_maintenance)
    def calculate_analytical_metrics(self):
        print(f"\n[DEBUG] Starting calculate_analytical_metrics for: {self.name}")
        if not ANALYTICAL_QUEUE_CONFIG.get("enabled") or not self.model_config:
            self.analytical_results = {"status": "disabled or not configured"}
            return
        model_type = self.model_config.get("type")
        model_func = ANALYTICAL_MODELS.get(model_type)
        if not model_func:
            self.analytical_results = {"error": f"Model type '{model_type}' not found."}
            return
        params = {}
        source = ANALYTICAL_QUEUE_CONFIG.get("parameter_source", "auto")
        if source == 'auto':
            if len(self.arrival_timestamps) < 2 or len(self.service_times) < 2:
                self.analytical_results = {"error": "Not enough data for auto-parameterization."}
                return
            def scv(data):
                if len(data) < 2 or np.mean(data) == 0: return 0.0
                return np.var(data) / (np.mean(data) ** 2)
            inter_arrival_times = np.diff(self.arrival_timestamps)
            mean_inter_arrival = np.mean(inter_arrival_times)
            params['lam'] = 1 / mean_inter_arrival if mean_inter_arrival > 0 else 0
            mean_service = np.mean(self.service_times)
            params['mu'] = 1 / mean_service if mean_service > 0 else 0
            params['arrival_scv'] = scv(inter_arrival_times)
            params['service_scv'] = scv(self.service_times)
            params['c'] = self.config.get('num_machines', 1)
        else: 
            params['lam'] = self.model_config.get('arrival_rate_lambda')
            params['mu'] = self.model_config.get('service_rate_mu')
            params['c'] = self.config.get('num_machines', 1)
            params['arrival_scv'] = self.model_config.get('arrival_scv', 1.0)
            params['service_scv'] = self.model_config.get('service_scv', 1.0)
        if any(p is None for p in [params.get('lam'), params.get('mu')]):
            self.analytical_results = {"error": "Manual parameters (lambda, mu) are not fully configured."}
            return
        self.analytical_params = {"source": source, **params}
        lam = params.get('lam', 0)
        mu = params.get('mu', 0)
        c = params.get('c', 1)
        print(f"    - Analyzing {self.name} ({model_type}):")
        print(f"      - Arrival Rate (λ): {lam:.4f} jobs/hr")
        print(f"      - Service Rate (μ): {mu:.4f} jobs/hr")
        print(f"      - Capacity (c): {c}")
        print(f"      - Total Capacity (c*μ): {c*mu:.4f} jobs/hr")
        if lam >= c*mu:
            print("      - [!!!] SYSTEM UNSTABLE: Arrival rate is greater than or equal to service capacity.")
        try:
            results = model_func(**params)
            cleaned_results = {}
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (np.float64, np.int64)):
                        cleaned_results[key] = float(value)
                    else:
                        cleaned_results[key] = value
            else:
                 cleaned_results = results
            self.analytical_results = cleaned_results
        except Exception as e:
            self.analytical_results = {"error": f"Calculation failed: {str(e)}"}
def machine_worker_process(env, stage_machines, worker, job_id, product_type, stage, metrics, inventory_manager=None, job_logger=None, worker_pool=None, order_id=None, ordertracker=None):
    import random as _rnd
    product_config = PRODUCTS[product_type]
    machine = pick_machine(stage_machines)
    current_process = env.active_process
    timeline = machine.timeline
    worker_event = worker_pool.request_worker(stage)
    assigned_worker = yield worker_event
    if assigned_worker:
        assigned_worker.assign_to_machine(machine, "production")
        efficiency_multiplier = assigned_worker.get_efficiency_for_machine(stage)
    else:
        efficiency_multiplier = 1.0
    if job_logger is not None:
        job_logger.log('stage_enter', env.now, job_id=job_id, product=product_type, stage=stage, 
                      machine_name=getattr(machine, 'name', ''))
    if inventory_manager:
        if ('transport_manager' in globals() or 'transport_manager' in locals()):
            try:
                _ptype = product_type if 'product_type' in locals() else (product if 'product' in locals() else None)
                if _ptype is not None:
                    _ptype = str(_ptype).strip()
                stage = str(stage).strip()
                _consumes_here = (_ptype is not None and 
                                  _ptype in MATERIAL_CONSUMPTION and 
                                  stage in MATERIAL_CONSUMPTION[_ptype] and 
                                  MATERIAL_CONSUMPTION[_ptype][stage])
            except Exception:
                _consumes_here = False
            if _consumes_here:
                _stage_name = stage if 'stage' in locals() else (stage_name if 'stage_name' in locals() else None)
                if not _stage_name:
                    _stage_name = str(getattr(self, 'stage_name', '')).strip() or str(stage)
                _buf = transport_manager.inbound_buffers.get(_stage_name)
                if _buf is not None:
                    if len(getattr(_buf, 'items', [])) == 0:
                        _evt = transport_manager.request_transport(_stage_name, qty_units=1, mode_hint="forklift")
                        try:
                            yield env.timeout(60.0)
                        except Exception:
                            pass
                    if len(getattr(_buf, 'items', [])) > 0:
                        yield _buf.get()
        if not inventory_manager.consume_materials(product_type, stage):
            if job_logger is not None:
                job_logger.log('material_wait_start', env.now, job_id=job_id, product=product_type, 
                              stage=stage, machine_name=getattr(machine, 'name', ''))
            print(f"BLOCKED: Job {job_id} cannot start {stage} due to material shortage")
            while True:
                if not calendar.is_working_time(env.now, stage_name=stage_name):
                    t1 = calendar.next_working_time(env.now, stage_name=stage_name)
                    timeline.add_interval(env.now, t1, 'calendar_off')
                    yield env.timeout(t1 - env.now)
                else:
                    t0 = env.now
                    yield env.timeout(1)
                    timeline.add_interval(t0, env.now, 'waiting_material')
                    if inventory_manager.consume_materials(product_type, stage):
                        print(f"RESUMED: Job {job_id} can now start {stage} - materials available")
                        if job_logger is not None:
                            job_logger.log('material_wait_end', env.now, job_id=job_id, product=product_type, 
                                          stage=stage, machine_name=getattr(machine, 'name', ''))
                        break
    yield from calendar_pause_until_working(env, stage, calendar, timeline)
    from queuing_system import Job 
    processing_time_config = product_config['machine_times'][stage]
    if isinstance(processing_time_config, (list, tuple)):
        time_estimate = processing_time_config[0] 
    else:
        time_estimate = float(processing_time_config)
    job_details = Job(
        job_id=job_id,
        product_type=product_type,
        stage=stage,
        arrival_time=env.now,
        process=env.active_process, 
        profit=product_config.get('profitperunit', 0),
        due_date=ordertracker.orders[order_id].get('duetime', float('inf')) if ordertracker and order_id and order_id in ordertracker.orders else float('inf'),
        processing_time_estimate=time_estimate
    )
    if job_logger:
        job_logger.log('stage_queued', env.now, job_id=job_id, stage=stage, machine_name=machine.name)
    timeline.set_state(env.now, 'waiting_in_queue')
    request_start_time = env.now
    yield machine.request(job_details)
    service_start_time = env.now
    wait_time = env.now - request_start_time 
    if job_logger:
        job_logger.log('stage_start', env.now, job_id=job_id, stage=stage, machine_name=machine.name, extra={'queue_wait_time': wait_time})
        try:
            stage_activity_start_ts = env.now
            actual_setup_time = 0
            if machine.last_product_type != product_type:
                actual_setup_time = product_config['setup_times'][stage] / efficiency_multiplier
                if actual_setup_time > 0:
                    if job_logger is not None:
                        job_logger.log('setup_start', env.now, job_id=job_id, product=product_type, 
                                      stage=stage, machine_name=getattr(machine, 'name', ''), 
                                      extra={'setup_time': actual_setup_time})
                    timeline.set_state(env.now, 'setup')
                    try:
                        yield from calendar_timeout(env, actual_setup_time, stage, calendar, timeline)
                    except simpy.Interrupt as interrupt:
                        if interrupt.cause == 'machine_down':
                            if job_logger is not None:
                                job_logger.log('interrupt', env.now, job_id=job_id, product=product_type, 
                                              stage=stage, machine_name=getattr(machine, 'name', ''), 
                                              extra={'phase': 'setup', 'remaining_time': actual_setup_time})
                            while not machine.is_available():
                                if not calendar.is_working_time(env.now, stage=stage):
                                    t1 = calendar.next_working_time(env.now, stage=stage)
                                    timeline.add_interval(env.now, t1, 'calendar_off')
                                    yield env.timeout(t1 - env.now)
                                else:
                                    yield env.timeout(0.5)
                            if job_logger is not None:
                                job_logger.log('resume', env.now, job_id=job_id, product=product_type, 
                                              stage=stage, machine_name=getattr(machine, 'name', ''), 
                                              extra={'phase': 'setup'})
                            yield from calendar_timeout(env, actual_setup_time, stage, calendar, timeline)
                        else:
                            raise
                    if job_logger is not None:
                        job_logger.log('setup_end', env.now, job_id=job_id, product=product_type, 
                                      stage=stage, machine_name=getattr(machine, 'name', ''))
                machine.last_product_type = product_type
            assigned_worker = None
            worker_req = None
            efficiency_multiplier = 1.0 
            if worker_pool and MACHINE_WORKER_REQUIREMENTS.get(stage, {}).get('requires_worker', True):
                timeline.set_state(env.now, 'waiting_worker')
                if job_logger:
                    job_logger.log('worker_wait_start', env.now, job_id=job_id, stage=stage)
                worker_req = worker_pool.request_worker(stage)
                assigned_worker = yield worker_req  
                if job_logger:
                    job_logger.log('worker_wait_end', env.now, job_id=job_id, stage=stage,
                                   extra={'worker_id': assigned_worker.worker_id if assigned_worker else "None"})
                if assigned_worker:
                    assigned_worker.assign_to_machine(machine, 'production')
                    efficiency_multiplier = assigned_worker.get_efficiency_for_machine(stage)
                else:
                    print(f"WARNING: Job {job_id} at stage {stage} could not get a worker.")
                    efficiency_multiplier = 1.0
            time_config = product_config['machine_times'][stage]
            if isinstance(time_config, str):
                import ast
                try:
                    time_config = ast.literal_eval(time_config)
                except (ValueError, SyntaxError):
                    time_config = float(time_config)
            if isinstance(time_config, (list, tuple)):
                machine_time = _rnd.uniform(float(time_config[0]), float(time_config[1])) / efficiency_multiplier
            else:
                machine_time = float(time_config) / efficiency_multiplier
            remaining_time = machine_time
            if job_logger is not None:
                job_logger.log('process_start', env.now, job_id=job_id, product=product_type, 
                              stage=stage, machine_name=getattr(machine, 'name', ''), 
                              extra={'process_time': machine_time})
            timeline.set_state(env.now, 'running')
            while remaining_time > 1e-9:
                try:
                    step = min(remaining_time, 1.0)
                    start_chunk = env.now
                    yield from calendar_timeout(env, step, stage, calendar, timeline)
                    elapsed = env.now - start_chunk
                    remaining_time -= elapsed
                except simpy.Interrupt as interrupt:
                    if interrupt.cause == 'machine_down':
                        if job_logger is not None:
                            job_logger.log('interrupt', env.now, job_id=job_id, product=product_type, 
                                          stage=stage, machine_name=getattr(machine, 'name', ''), 
                                          extra={'phase': 'processing', 'remaining_time': remaining_time})
                        while not machine.is_available():
                            if not calendar.is_working_time(env.now, stage=stage):
                                t1 = calendar.next_working_time(env.now, stage=stage)
                                timeline.add_interval(env.now, t1, 'calendar_off')
                                yield env.timeout(t1 - env.now)
                            else:
                                yield env.timeout(0.5)
                        if job_logger is not None:
                            job_logger.log('resume', env.now, job_id=job_id, product=product_type, 
                                          stage=stage, machine_name=getattr(machine, 'name', ''), 
                                          extra={'phase': 'processing', 'remaining_time': remaining_time})
                        timeline.set_state(env.now, 'running')
                    else:
                        raise
            if job_logger is not None:
                job_logger.log('process_end', env.now, job_id=job_id, product=product_type, 
                              stage=stage, machine_name=getattr(machine, 'name', ''))
            if job_logger is not None:
                job_logger.log('worker_wait_start', env.now, job_id=job_id, product=product_type, 
                              stage=stage, machine_name=getattr(machine, 'name', ''), 
                              extra={'phase': 'unloading'})
            timeline.set_state(env.now, 'waiting_worker')
            with worker.request() as wreq2:
                yield wreq2
                if worker_pool is not None:
                    worker_pool.in_use[stage] = worker_pool.in_use.get(stage, 0) + 1
                    try:
                        rt_metrics.set_gauge("worker_utilization", worker_pool.in_use[stage], env.now)
                    except Exception:
                        pass
                try:
                    if 'worker_pool' in globals() and worker_pool is not None:
                        total = len(worker_pool.workers)
                        working = sum(1 for w in worker_pool.workers.values() if w.is_working)
                        rt_metrics.observe('worker_utilization', working / total, env.now)
                except Exception:
                    pass
                try:
                    if job_logger is not None:
                        job_logger.log('worker_wait_end', env.now, job_id=job_id, product=product_type, 
                                      stage=stage, machine_name=getattr(machine, 'name', ''), 
                                      extra={'phase': 'unloading'})
                    unload_config = product_config.get('unload_times', {}).get(stage, 0.1)
                    if isinstance(unload_config, str):
                        import ast
                        try:
                            unload_config = ast.literal_eval(unload_config)
                        except (ValueError, SyntaxError):
                            unload_config = float(unload_config)
                    if isinstance(unload_config, (list, tuple)):
                        unload_time = _rnd.uniform(float(unload_config[0]), float(unload_config[1])) / efficiency_multiplier
                    else:
                        unload_time = float(unload_config) / efficiency_multiplier
                    if job_logger is not None:
                        job_logger.log('unload_start', env.now, job_id=job_id, product=product_type, 
                                      stage=stage, machine_name=getattr(machine, 'name', ''), 
                                      extra={'unload_time': unload_time})
                    timeline.set_state(env.now, 'setup')
                    try:
                        yield from calendar_timeout(env, unload_time, stage, calendar, timeline)
                    except simpy.Interrupt as interrupt:
                        if interrupt.cause == 'machine_down':
                            if job_logger is not None:
                                job_logger.log('interrupt', env.now, job_id=job_id, product=product_type, 
                                              stage=stage, machine_name=getattr(machine, 'name', ''), 
                                              extra={'phase': 'unloading', 'remaining_time': unload_time})
                            while not machine.is_available():
                                if not calendar.is_working_time(env.now, stage=stage):
                                    t1 = calendar.next_working_time(env.now, stage=stage)
                                    timeline.add_interval(env.now, t1, 'calendar_off')
                                    yield env.timeout(t1 - env.now)
                                else:
                                    yield env.timeout(0.5)
                            if job_logger is not None:
                                job_logger.log('resume', env.now, job_id=job_id, product=product_type, 
                                              stage=stage, machine_name=getattr(machine, 'name', ''), 
                                              extra={'phase': 'unloading'})
                            yield from calendar_timeout(env, unload_time, stage, calendar, timeline)
                        else:
                            raise
                    if job_logger is not None:
                        job_logger.log('unload_end', env.now, job_id=job_id, product=product_type, 
                                      stage=stage, machine_name=getattr(machine, 'name', ''))
                finally:
                    if worker_pool is not None:
                        worker_pool.in_use[stage] = max(0, worker_pool.in_use.get(stage, 1) - 1)
                        try:
                            rt_metrics.set_gauge("worker_utilization", worker_pool.in_use[stage], env.now)
                        except Exception:
                            pass
                        try:
                            if 'worker_pool' in globals() and worker_pool is not None:
                                total = len(worker_pool.workers)
                                working = sum(1 for w in worker_pool.workers.values() if w.is_working)
                                rt_metrics.observe('worker_utilization', working / total, env.now)
                        except Exception:
                            pass
            finish_time = env.now
            m = metrics[stage]
            m['start_times'].append((job_id, product_type, stage_activity_start_ts))
            m['completion_times'].append((job_id, product_type, finish_time))
            m['stage_flow_times'].append((job_id, product_type, finish_time - stage_activity_start_ts))
            print(f"Job {job_id} ({product_type}) finished {stage} at {finish_time:.2f} on {getattr(machine, 'name', 'stage')}")
            if job_logger is not None:
                job_logger.log('stage_exit', env.now, job_id=job_id, product=product_type, 
                              stage=stage, machine_name=getattr(machine, 'name', ''), 
                              extra={'stage_flow_time': finish_time - stage_activity_start_ts})
            timeline.set_state(env.now, 'idle')
            qc_present = False
            try:
                stage_req = MACHINE_WORKER_REQUIREMENTS.get(stage, {})
                if bool(stage_req.get('quality_critical', False)):
                    qc_present = True
                if ('assigned_worker' in locals()) and (assigned_worker is not None):
                    if hasattr(assigned_worker, 'skills') and ('qualitycontrol' in assigned_worker.skills):
                        qc_present = True
            except Exception:
                qc_present = False
            if qualitytracker is not None:
                outcome = qualitytracker.estimate_defect_outcome(product_type, stage, qc_present)
                detected_defect = False
                did_rework = False
                was_scrap = False
                escaped = False
                if outcome['occurs']:
                    if outcome['detected']:
                        detected_defect = True
                        rw_enabled, rw_max, rw_factor, rw_success_p = qualitytracker.rework_decision()
                        if rw_enabled:
                            import random
                            loops = 0
                            base_duration = max(0.0, finish_time - stage_activity_start_ts)
                            rework_time = max(0.0, base_duration * rw_factor)
                            success = True
                            while loops < rw_max:
                                did_rework = True
                                try:
                                    yield from calendar_timeout(env, rework_time, stage, calendar, timeline)
                                except simpy.Interrupt:
                                    pass
                                if random.random() < rw_success_p:
                                    success = True
                                    break
                                else:
                                    success = False
                                    loops += 1
                            if not success:
                                was_scrap = True
                                if financial_tracker:
                                    financial_tracker.total_costs += qualitytracker.scrap_cost_for(product_type)
                                qualitytracker.record_stage_result(
                                    stage=stage,
                                    product=product_type,
                                    duration_h=base_duration,
                                    qc_present=qc_present,
                                    detected_defect=True,
                                    reworked=True,
                                    scrap=True,
                                    escaped=False
                                )
                                return
                            else:
                                qualitytracker.record_stage_result(
                                    stage=stage,
                                    product=product_type,
                                    duration_h=base_duration,
                                    qc_present=qc_present,
                                    detected_defect=True,
                                    reworked=True,
                                    scrap=False,
                                    escaped=False
                                )
                        else:
                            was_scrap = True
                            if financial_tracker:
                                financial_tracker.total_costs += qualitytracker.scrap_cost_for(product_type)
                            base_duration = max(0.0, finish_time - stage_activity_start_ts)
                            qualitytracker.record_stage_result(
                                stage=stage,
                                product=product_type,
                                duration_h=base_duration,
                                qc_present=qc_present,
                                detected_defect=True,
                                reworked=False,
                                scrap=True,
                                escaped=False
                            )
                            return
                    else:
                        escaped = True
                        qualitytracker.record_escape(product_type, 1)
                        base_duration = max(0.0, finish_time - stage_activity_start_ts)
                        qualitytracker.record_stage_result(
                            stage=stage,
                            product=product_type,
                            duration_h=base_duration,
                            qc_present=qc_present,
                            detected_defect=False,
                            reworked=False,
                            scrap=False,
                            escaped=True
                        )
                else:
                    base_duration = max(0.0, finish_time - stage_activity_start_ts)
                    qualitytracker.record_stage_result(
                        stage=stage,
                        product=product_type,
                        duration_h=base_duration,
                        qc_present=qc_present,
                        detected_defect=False,
                        reworked=False,
                        scrap=False,
                        escaped=False
                    )
        finally:
            if 'assigned_worker' in locals() and assigned_worker:
                print(f"RELEASING WORKER: {assigned_worker.name} at {env.now:.2f}")
                assigned_worker.release_from_machine()
                worker_pool.release_worker(assigned_worker.worker_id, machine)
            else:
                print(f"NO WORKER TO RELEASE at {env.now:.2f}")
def job_process(env, job_id, product_type, machines, workers, metrics, job_metrics, financial_tracker, inventory_manager, order_tracker, order_id, item_id, job_logger, worker_pool, PRODUCTS, MATERIAL_CONSUMPTION,cfg,environmental=None, qualitytracker=None, calendar=None, conwip_container=None):
    import random as _rnd
    import ast
    product_config = PRODUCTS[product_type]
    routing = product_config['routing']
    job_start = env.now
    machines_used = []
    print(f"DEBUG: qualitytracker is {'None' if qualitytracker is None else 'available'}")
    if job_logger is not None:
        job_logger.log('job_started', env.now, job_id=job_id, order_id=order_id, item_id=item_id, product=product_type)
    for stage in routing:
        assigned_worker = None  
        efficiency_multiplier = 1.0
        try:
            attempt = 0
            while True:
                attempt += 1
                yield from calendar_pause_until_working(env, stage, calendar, None) 
                machine = pick_machine(machines[stage])
                timeline = machine.timeline
                machines_used.append(machine.name)
                if worker_pool and MACHINE_WORKER_REQUIREMENTS.get(stage, {}).get('requires_worker', True):
                    timeline.set_state(env.now, 'waiting_worker')
                    if job_logger:
                        job_logger.log('worker_wait_start', env.now, job_id=job_id, stage=stage)
                    worker_req = worker_pool.request_worker(stage)
                    assigned_worker = yield worker_req
                    if assigned_worker:
                        assigned_worker.assign_to_machine(machine, 'production')
                        efficiency_multiplier = assigned_worker.get_efficiency_for_machine(stage)
                        if job_logger:
                            job_logger.log('worker_wait_end', env.now, job_id=job_id, stage=stage, extra={'worker_id': assigned_worker.worker_id})
                    else:
                        if job_logger:
                             job_logger.log('worker_wait_end', env.now, job_id=job_id, stage=stage, extra={'worker_id': "None"})
                        print(f"WARNING: Job {job_id} could not get a worker for stage {stage}.")
                if inventory_manager:
                    materials_needed = MATERIAL_CONSUMPTION.get(product_type, {}).get(stage, {})
                    if not materials_needed:
                        print(f"INFO: No materials defined for product '{product_type}' at stage '{stage}'.")
                    else:
                        while not inventory_manager.consume_materials(materials_needed, quantity=1, product_type=product_type, stage=stage):
                            timeline.set_state(env.now, 'waiting_material')
                            if job_logger:
                                job_logger.log('material_wait_start', env.now, job_id=job_id, stage=stage)
                            yield env.timeout(1)
                        if job_logger:
                            job_logger.log('material_wait_end', env.now, job_id=job_id, stage=stage)
                        if environmental:
                            for mat, amt in materials_needed.items():
                                environmental.record_material_consumption(mat, amt, product_type, stage)
                from queuing_system import Job  
                processing_time_config = product_config['machine_times'][stage]
                if isinstance(processing_time_config, (list, tuple)):
                    time_estimate = (float(processing_time_config[0]) + float(processing_time_config[1])) / 2.0
                else:
                    time_estimate = float(processing_time_config)
                unit_price = product_config.get('unit_price', 0)
                material_cost = 0
                materials_needed = MATERIAL_CONSUMPTION.get(product_type, {}).get(stage, {})
                for mat, qty in materials_needed.items():
                    material_cost += cfg.RAW_MATERIALS[mat]['cost_per_unit'] * qty
                job_profit = unit_price - material_cost
                job_due_date = float('inf')
                if order_tracker and order_id in order_tracker.orders:
                    job_due_date = order_tracker.orders[order_id].get('due_time') or float('inf')
                job_context = {
                    "machines": machines,
                    "workers": workers,
                    "metrics": metrics,
                    "job_metrics": job_metrics,
                    "financial_tracker": financial_tracker,
                    "inventory_manager": inventory_manager,
                    "order_tracker": order_tracker,
                    "job_logger": job_logger,
                    "worker_pool": worker_pool,
                    "PRODUCTS": PRODUCTS,
                    "MATERIAL_CONSUMPTION": MATERIAL_CONSUMPTION,
                    "cfg": cfg,
                    "environmental": environmental,
                    "qualitytracker": qualitytracker,
                    "calendar": calendar,
                    "conwip_container": conwip_container
                }
                job_details = Job(
                    job_id=job_id,
                    product_type=product_type,
                    stage=stage,
                    arrival_time=env.now,
                    process=env.active_process,
                    profit=job_profit,  
                    due_date=job_due_date,  
                    processing_time_estimate=time_estimate,  
                    routing=routing,
                    context=job_context
                )
                print(f"--> Time {env.now:.2f}: ENQUEUE job '{job_details.job_id}' for '{stage}'. Est. Time: {job_details.processing_time_estimate:.2f}h, Profit: ${job_details.profit:.2f}, Due: {job_details.due_date or 'N/A'}")
                timeline.set_state(env.now, 'waiting_in_queue')
                if job_logger:
                    job_logger.log('stage_queued', env.now, job_id=job_id, stage=stage, machine_name=machine.name)
                request_event = machine.request(job_details)
                result = yield request_event 
                if result and result.get('status') == 'rework':
                    continue
                elif result and result.get('status') == 'scrapped':
                    break
                stage_activity_start_ts = env.now
                actual_setup_time = 0
                if machine.last_product_type != product_type:
                    seq_setup_time = None
                    if machine.last_product_type and machine.last_product_type != product_type:
                        seq_setup_time = cfg.SETUP_MATRIX.get(machine.last_product_type, {}).get(product_type)
                    if seq_setup_time is not None:
                        print(f"DEBUG: Overriding setup for {product_type} from {machine.last_product_type}. Using sequence time: {seq_setup_time:.2f}h")
                        actual_setup_time = seq_setup_time / efficiency_multiplier
                    else:
                        actual_setup_time = product_config['setup_times'][stage] / efficiency_multiplier
                    if actual_setup_time > 0:
                        timeline.set_state(env.now, 'setup')
                        yield from calendar_timeout(env, actual_setup_time, stage, calendar, timeline)
                machine.last_product_type = product_type
                time_config = product_config['machine_times'][stage]
                if isinstance(time_config, str):
                    import ast
                    try:
                        time_config = ast.literal_eval(time_config)
                    except (ValueError, SyntaxError):
                        time_config = float(time_config)
                if isinstance(time_config, (list, tuple)):
                    machine_time = _rnd.uniform(float(time_config[0]), float(time_config[1])) / efficiency_multiplier
                else:
                    machine_time = float(time_config) / efficiency_multiplier
                timeline.set_state(env.now, 'running')
                service_start_time = env.now
                yield from calendar_timeout(env, machine_time, stage, calendar, timeline)
                service_duration = env.now - service_start_time
                if ANALYTICAL_QUEUE_CONFIG.get("enabled") and service_duration > 0:
                    machine.service_times.append(service_duration)
                unload_config = product_config.get('unload_times', {}).get(stage, 0.1)
                if isinstance(unload_config, str):
                    import ast
                    try:
                        unload_config = ast.literal_eval(unload_config)
                    except (ValueError, SyntaxError):
                        unload_config = float(unload_config)
                if isinstance(unload_config, (list, tuple)):
                    unload_time = _rnd.uniform(float(unload_config[0]), float(unload_config[1])) / efficiency_multiplier
                else:
                    unload_time = float(unload_config) / efficiency_multiplier
                timeline.set_state(env.now, 'setup') 
                yield from calendar_timeout(env, unload_time, stage, calendar, timeline)
                if qualitytracker:
                    print(f"DEBUG: Quality check for job {job_id}, stage {stage}")
                    outcome = qualitytracker.estimate_defect_outcome(product_type, stage, qc_present=True)
                    print(f"DEBUG: Quality outcome: {outcome}")
                    base_duration = max(0.0, env.now - stage_activity_start_ts)
                    if outcome['occurs']:
                        if outcome['detected']:
                            qualitytracker.record_stage_result(
                                stage=stage, product=product_type, duration_h=base_duration,
                                qc_present=True, detected_defect=True, reworked=True, 
                                scrap=False, escaped=False
                            )
                        else:
                            qualitytracker.record_stage_result(
                                stage=stage, product=product_type, duration_h=base_duration,
                                qc_present=True, detected_defect=False, reworked=False, 
                                scrap=False, escaped=True
                            )
                    else:
                        qualitytracker.record_stage_result(
                            stage=stage, product=product_type, duration_h=base_duration,
                            qc_present=True, detected_defect=False, reworked=False, 
                            scrap=False, escaped=False
                        )
                if environmental:
                    environmental.record_transport(mode='forklift', km=0.1, hours=0.1)
                timeline.set_state(env.now, 'idle')
                machine.record_unit_produced()
                if job_logger is not None:
                    finish_time = env.now
                    job_logger.log(
                        'stage_exit', 
                        finish_time, 
                        job_id=job_id, 
                        product=product_type,
                        stage=stage, 
                        machine_name=getattr(machine, 'name', stage),
                        extra={'stage_flow_time': finish_time - stage_activity_start_ts}
                    )
                break 
        finally:
            if assigned_worker and worker_pool:
                worker_pool.release_worker(assigned_worker.worker_id, machine)
                if job_logger:
                    job_logger.log('worker_released', env.now, job_id=job_id, stage=stage, extra={'worker_id': assigned_worker.worker_id})
    job_finish = env.now
    cycle_time = job_finish - job_start
    job_metrics[job_id] = {'start': job_start, 'finish': job_finish, 'cycle_time': cycle_time, 'product_type': product_type}
    financial_tracker.record_production_completion(job_id, product_type, job_finish, cycle_time, list(set(machines_used)))
    inventory_manager.receive_finished_unit(product_type, quantity=1)
    order_tracker.mark_item_completed(order_id, item_id)
    if job_logger is not None:
        job_logger.log('job_completed', env.now, job_id=job_id, order_id=order_id, item_id=item_id, product=product_type, cycle_time=cycle_time)
    if conwip_container:
        yield conwip_container.put(1)
        print(f"--- Time {env.now:.2f}: CONWIP slot released by job '{job_id}'.")
def order_processor(env, orders, machines, workers, metrics, job_metrics,
                    financial_tracker, inventory_manager, order_tracker, job_logger, worker_pool, 
                    PRODUCTS, MATERIAL_CONSUMPTION, cfg, environmental, qualitytracker, calendar,conwip_container=None):
    print(f"DEBUG in order_processor: Calendar object received: {calendar is not None}")
    if calendar is None:
        import sys
        print("FATAL: order_processor did not receive the calendar object. Aborting.", file=sys.stderr)
        return
    job_id_counter = 1
    for order in orders:
        order_id = order['order_id']
        due_in = order.get('due_in_hours', None)
        order_tracker.register_order(order_id, order['products'], release_time=env.now, due_in_hours=due_in)
        job_logger.log('order_released', env.now, order_id=order_id)
        print(f"--- Time {env.now:.2f}: Processing Order {order_id} ---")
        for idx, product_type in enumerate(order['products'], start=1):
            if conwip_container:
                print(f"--> Time {env.now:.2f}: Order processor waiting for CONWIP slot...")
                yield conwip_container.get(1)
                print(f"+++ Time {env.now:.2f}: CONWIP slot acquired. Releasing job.")
            if product_type in PRODUCTS:
                item_id = f"{order_id}_item_{idx}"
                env.process(job_process(env, job_id_counter, product_type, machines, workers, 
                                     metrics, job_metrics, financial_tracker, inventory_manager, order_tracker, 
                                     order_id, item_id, job_logger, worker_pool, PRODUCTS, MATERIAL_CONSUMPTION, 
                                     cfg, environmental, qualitytracker, calendar))
                job_id_counter += 1
        if 'next_order_arrival_delay' in order:
             yield env.timeout(order['next_order_arrival_delay'])
def daily_shipping_process(env, order_tracker, inventory_manager, financial_tracker, qualitytracker=None, calendar=None):
    daily_hour = SHIPPING_CONFIG.get('daily_shipping_hour', 23) % 24.0
    def hours_to_next_daily(hour_now):
        _, h = calendar._day_hour(hour_now)
        delta = (daily_hour - h) % 24.0
        return delta if delta > 1e-6 else 24.0
    if calendar:
        first_wait = hours_to_next_daily(env.now)
        if first_wait > 0:
            yield env.timeout(first_wait)
    while True:
        print(f"SHIPPING WINDOW OPEN at t={env.now:.2f}")
        order_ids_at_window_open = list(order_tracker.orders.keys())
        for oid in order_ids_at_window_open:
            od = order_tracker.orders[oid] 
            if od['status'] == 'completed':
                shipped_units = inventory_manager.ship_available_for_order(od)
                if shipped_units > 0:
                    order_tracker.mark_order_shipped(oid)
                    print(f"Order {oid} shipped ({shipped_units} units) at t={env.now:.2f}")
                    try:
                        _qt = qualitytracker if 'qualitytracker' in globals() or 'qualitytracker' in locals() else quality_tracker
                    except NameError:
                        _qt = quality_tracker if 'quality_tracker' in globals() or 'quality_tracker' in locals() else None
                    if _qt is not None and (shipped_units if 'shipped_units' in locals() else shippedunits) > 0:
                        _oid = oid if 'oid' in locals() else order_id
                        _ot = ordertracker if 'ordertracker' in locals() else order_tracker
                        _products_cfg = PRODUCTS
                        product_mix = {}
                        try:
                            odata = _ot.orders[_oid]
                            for it in odata.get('items', []):
                                p = it.get('product')
                                if p:
                                    product_mix[p] = product_mix.get(p, 0) + 1
                        except Exception:
                            pass
                        total_shipped = shipped_units if 'shipped_units' in locals() else shippedunits
                        if not product_mix:
                            try:
                                first_product = next(iter(_products_cfg.keys()))
                                product_mix = {first_product: total_shipped}
                            except StopIteration:
                                product_mix = {}
                        total_in_mix = sum(product_mix.values()) or 1
                        remaining = total_shipped
                        for p, cnt in product_mix.items():
                            qty_est = int(round(total_shipped * (cnt / total_in_mix)))
                            qty = qty_est if qty_est <= remaining else remaining
                            remaining -= qty
                            if qty <= 0:
                                continue
                            suspect = min(qty, _qt.escapes_by_product.get(p, 0))
                            good = max(0, qty - suspect)
                            _qt.record_shipment(p, good, suspect)
                            if suspect > 0:
                                env.process(_qt.schedule_potential_returns(_oid, p, suspect))
        yield env.timeout(24.0)
def static_sequence_scheduler(
    env, master_sequence: List[str], all_orders: List[Dict],
    machines, worker_pool, rt_metrics, job_logger,
    financial_tracker, inventory_manager, order_tracker,
    environmental, qualitytracker, calendar, config_ns, conwip_container=None):
    print(f"\n--- EXECUTING STATIC MASTER PLAN of {len(master_sequence)} jobs ---")
    orders_by_id = {order['id']: order for order in all_orders}
    job_metrics = {} 
    for job_id_to_release in master_sequence:
        if job_id_to_release in orders_by_id:
            order_data = orders_by_id[job_id_to_release]
            product_type = order_data['products'][0]
            order_tracker.register_order(
                order_id=job_id_to_release,
                products=[product_type],
                release_time=env.now,
                due_in_hours=order_data.get('due_in_hours')
            )
            item_id = order_tracker.orders[job_id_to_release]['items'][0]['item_id']
            env.process(job_process(
                env=env,
                job_id=f"Job_{job_id_to_release}",
                product_type=product_type,
                machines=machines,
                workers=locals().get('workers'), 
                metrics=rt_metrics,
                job_metrics=job_metrics, 
                financial_tracker=financial_tracker,
                inventory_manager=inventory_manager,
                order_tracker=order_tracker,
                order_id=job_id_to_release,
                item_id=item_id,
                job_logger=job_logger,
                worker_pool=worker_pool,
                PRODUCTS=config_ns.PRODUCTS,
                MATERIAL_CONSUMPTION=config_ns.MATERIAL_CONSUMPTION,
                cfg=config_ns,
                environmental=environmental,
                qualitytracker=qualitytracker,
                calendar=calendar,
                conwip_container=conwip_container
            ))
        else:
            print(f"WARNING (Static Scheduler): Job ID '{job_id_to_release}' from master sequence not found in order list.")
def csv_orders_scheduler(env, csv_orders, machines, workers, metrics, job_metrics,
                         financial_tracker, inventory_manager, ordertracker, job_logger,
                         worker_pool, environmental, qualitytracker, calendar, config_ns,
                         conwip_container=None):
    def _order_releaser(order_data):
        release_time = float(order_data.get('release_time', 0.0) or 0.0)
        order_id = order_data.get('id') or order_data.get('order_id')
        if release_time > env.now:
            yield env.timeout(release_time - env.now)
        print(f"--- Time {env.now:.2f}: Releasing Order {order_id} ---")
        order_id = order_data.get('id') or order_data.get('order_id')
        if not order_id:
            order_id = f"ORD_{int(env.now)}_{int(release_time)}"
        products_raw = order_data.get('products', [])
        products = []
        if products_raw is None:
            products_raw = []
        elif isinstance(products_raw, str):
            try:
                loaded = json.loads(products_raw)
                if isinstance(loaded, list):
                    products_raw = loaded
                else:
                    products_raw = [str(loaded)]
            except (json.JSONDecodeError, TypeError):
                products_raw = [products_raw]
        for item in products_raw:
            if item is not None:
                cleaned_item = str(item).strip().replace('\u200b', '')
                if cleaned_item:
                    products.append(cleaned_item)
        if not products:
            print(f"WARNING: Order '{order_id}' at time {env.now:.2f} has no valid products after cleaning. Skipping.")
            return 
        due_in = order_data.get('due_in_hours', None)
        try:
            due_in = float(due_in) if due_in is not None and due_in != '' else None
        except (ValueError, TypeError):
            due_in = None
        ordertracker.register_order(
            order_id=order_id,
            products=products,
            release_time=env.now,
            due_in_hours=due_in
        )
        local_job_id_counter = 1
        for item in ordertracker.orders[order_id]['items']:
            product_type = item['product']
            item_id = item['item_id']
            if product_type in config_ns.PRODUCTS:
                if conwip_container:
                    yield conwip_container.get(1) 
                env.process(job_process(
                    env, f"Job_{order_id}_{local_job_id_counter}", product_type, machines, workers, metrics, job_metrics,
                    financial_tracker, inventory_manager, ordertracker, order_id, item_id,
                    job_logger, worker_pool, config_ns.PRODUCTS, config_ns.MATERIAL_CONSUMPTION, config_ns,
                    environmental, qualitytracker, calendar, conwip_container
                ))
                local_job_id_counter += 1
            else:
                print(f"WARNING: Product '{product_type}' from order '{order_id}' not found in PRODUCTS configuration. Skipping item.")
    for order in csv_orders:
        env.process(_order_releaser(order))
def ensure_worker_utilization_tracking(worker_pool, rt_metrics, env_time):
    if worker_pool is None or rt_metrics is None:
        return 0.0
    try:
        total_workers = max(1, len(getattr(worker_pool, 'workers', {})))
        working_workers = sum(1 for w in worker_pool.workers.values() 
                            if getattr(w, 'is_working', False))
        current_util = working_workers / total_workers
        rt_metrics.observe('worker_utilization', current_util, env_time)
        return current_util
    except Exception as e:
        print(f"Error tracking worker utilization: {e}")
        return 0.0
def generate_csv_from_demand(csv_path, demand_config):
    print("--- Generating MULTI-PRODUCT orders from demand model ---")
    horizon = demand_config.get("horizon_days", 90)
    base_demand = demand_config.get("base_per_day", 10)
    model = demand_config.get("model", "seasonal_noise")
    seed = demand_config.get("rng_seed", None)
    product_mix = demand_config.get("product_mix", {})
    if not product_mix:
        product_mix = {demand_config.get("product_name", "ProductA"): 1.0}
    product_names = list(product_mix.keys())
    product_probabilities = list(product_mix.values())
    if seed is not None:
        np.random.seed(seed)
    gen = DemandGenerator()
    _, actuals = gen.generate_demand(horizon=horizon, base_demand=base_demand, model=model)
    orders = []
    order_counter = 1
    for day_idx, total_daily_qty_float in enumerate(actuals):
        total_daily_qty = max(0, int(round(total_daily_qty_float)))
        if total_daily_qty == 0:
            continue
        daily_products = np.random.choice(product_names, size=total_daily_qty, p=product_probabilities)
        release_time = (day_idx + 1) * 24 
        for product_name in daily_products:
            orders.append({
                "id": f"D{day_idx+1:03d}_{order_counter}",
                "release_time": float(release_time),
                "products": [product_name], 
                "due_in_hours": float(demand_config.get("due_lag_days", 7) * 24)
            })
            order_counter += 1
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "release_time", "products", "due_in_hours"])
        for row in orders:
            writer.writerow([
                row["id"],
                row["release_time"],
                json.dumps(row["products"]),
                row.get("due_in_hours", "")
            ])
    print(f"Successfully generated {len(orders)} multi-product orders into '{csv_path}'")
    return orders, csv_path
def run_simulation(config_ns, output_dir, scenario_label):
    if output_dir is None:
        scenario_name = scenario_label or 'baseline'
        output_dir = f"./out/{scenario_name}_{int(time.time())}"
        print(f"No output directory specified. Using default: {output_dir}")
    from config import METRICS_CONFIG
    cfg = config_ns if config_ns is not None else __import__("config")
    random.seed(config_ns.RANDOM_SEED)
    np.random.seed(config_ns.RANDOM_SEED)
    all_stages_set = set()
    for prod_config in cfg.PRODUCTS.values():
        all_stages_set.update(prod_config['routing'])
    DYNAMIC_STAGES = sorted(list(all_stages_set))
    print(f"Discovered {len(DYNAMIC_STAGES)} stages from product routings: {DYNAMIC_STAGES}")
    env = simpy.Environment()
    planning_data = PlanningData()
    conwip_container = None
    if cfg.FLOW_CONTROL_CONFIG.get('enabled', False):
        wip_limit = cfg.FLOW_CONTROL_CONFIG['wip_limit']
        print(f"PULL SYSTEM ENABLED: CONWIP level set to {wip_limit}")
        conwip_container = simpy.Container(env, capacity=wip_limit, init=wip_limit)
    else:
        print("PUSH SYSTEM ENABLED: No WIP limit in place.")
    try:
        from logging_export import JobLogger
        job_logger = JobLogger()
    except Exception:
        class _NullLogger:
            def __init__(self): self.events = []
            def log(self, *a, **k): pass
        job_logger = _NullLogger()
    environmental = EnvironmentalTracker(
        env,
        cfg.MACHINE_DEFINITIONS,
        cfg.ENVIRONMENTCONFIG,
        cfg.MACHINE_ENERGY_PROFILES,
        cfg.TRANSPORT_ENV,
        cfg.PRODUCT_ENV,
        cfg.MATERIAL_ENV
    )
    financial_tracker = FinancialTracker(env, cfg.MACHINE_DEFINITIONS, cfg.FINANCIAL_CONFIG, cfg.PRODUCTS, environmental=environmental)
    inventory_manager = InventoryManager(env,cfg, financial_tracker, cfg.RAW_MATERIALS, cfg.INVENTORY_POLICY, environmental=environmental, job_logger=job_logger, planning_data=planning_data)
    order_tracker = OrderTracker(env)
    qualitytracker = QualityTracker(env, cfg.QUALITYCONFIG, financial=financial_tracker, environmental=environmental)
    globals()["qualitytracker"] = qualitytracker
    globals()["financial_tracker"] = financial_tracker
    globals()["order_tracker"] = order_tracker
    globals()["environmental"] = environmental
    globals()["inventory_manager"] = inventory_manager
    if hasattr(cfg, 'SUPPLY_CHAIN_CONFIG') and cfg.SUPPLY_CHAIN_CONFIG.get('enabled', False):
        print("--- Supply Chain Simulation Enabled ---")
        supply_chain_manager = SupplyChainManager(
            env, 
            inventory_manager, 
            financial_tracker, 
            cfg.SUPPLY_CHAIN_CONFIG,  
            cfg.INVENTORY_POLICY,
            storage_capacity=cfg.INVENTORY_POLICY.get('storage_capacity')
        )
        env.process(supply_chain_manager.inventory_monitoring_process())
    else:
        print("--- Supply Chain Simulation Disabled ---")
    rt_metrics = MetricsBus()
    globals()["rt_metrics"] = rt_metrics 
    calendar = ShiftCalendar(cfg.CALENDAR_CONFIG)
    globals()["calendar"] = calendar
    worker_pool = WorkerPool(env, calendar, financial_tracker, cfg.WORKER_DEFINITIONS, rt_metrics)
    globals()["worker_pool"] = worker_pool
    try:
        from logging_export import JobLogger
        job_logger = JobLogger()
    except Exception:
        class _NullLogger:
            def __init__(self):
                self.events = []
            def log(self, *a, **k):
                pass
        job_logger = _NullLogger()
    globals()["job_logger"] = job_logger
    def _feed_environment_from_timelines(machines_map, environmental_tracker):
        for stage_name, m in machines_map.items():
            if isinstance(m, list):
                for mi in m:
                    if hasattr(mi, "timeline"):
                        mi.timeline.compress()
                        for s, e, lab in getattr(mi.timeline, "intervals", []):
                            environmental_tracker.record_machine_state_block(getattr(mi, "name", ""), stage_name, lab, s, e)
            else:
                if hasattr(m, "timeline"):
                    m.timeline.compress()
                    for s, e, lab in getattr(m.timeline, "intervals", []):
                        environmental_tracker.record_machine_state_block(getattr(m, "name", ""), stage_name, lab, s, e)
    machines = {}
    workers = {}
    globals()["machines"] = machines
    globals()["workers"] = workers
    def create_machine_from_config(env_, stage_name, MACHINE_DEFINITIONS, financial_tracker, is_worker=False, index=None):
        cfg_stage = MACHINE_DEFINITIONS.get(stage_name, cfg.DEFAULT_MACHINE_CONFIG)
        suffix = 'Worker' if is_worker else 'Machine'
        machine_name = f"{stage_name}_{suffix}{'' if index is None else index+1}"
        if is_worker:
            return EnhancedMachine(
                env_,
                name=machine_name,
                stage=stage_name,
                capacity=cfg_stage.get('capacity', 1),
                mtbf=cfg_stage.get('mtbf', cfg.DEFAULT_MACHINE_CONFIG['mtbf']),
                mttr=cfg_stage.get('mttr', cfg.DEFAULT_MACHINE_CONFIG['mttr']),
                maintenance_interval=cfg_stage.get('maintenance_interval', cfg.DEFAULT_MACHINE_CONFIG['maintenance_interval']),
                financial_tracker=financial_tracker,
                metrics_bus=rt_metrics,
                cfg=cfg,
                machines=machines,
                failure_model=cfg_stage.get('failure_model', 'exponential'),
                weibull_k=cfg_stage.get('weibull_k'),
                weibull_lambda=cfg_stage.get('weibull_lambda'),
                lognorm_mu=cfg_stage.get('lognorm_mu'),
                lognorm_sigma=cfg_stage.get('lognorm_sigma')
            )
        else:
            return EnhancedMachine(
                env_,
                name=machine_name,
                stage=stage_name,
                capacity=cfg_stage.get('capacity', 1),
                mtbf=cfg_stage.get('mtbf', cfg.DEFAULT_MACHINE_CONFIG['mtbf']),
                mttr=cfg_stage.get('mttr', cfg.DEFAULT_MACHINE_CONFIG['mttr']),
                maintenance_interval=cfg_stage.get('maintenance_interval', cfg.DEFAULT_MACHINE_CONFIG['maintenance_interval']),
                financial_tracker=financial_tracker,
                metrics_bus=rt_metrics,
                cfg=cfg,
                machines=machines,
                failure_model=cfg_stage.get('failure_model', 'exponential'),
                weibull_k=cfg_stage.get('weibull_k'),
                weibull_lambda=cfg_stage.get('weibull_lambda'),
                lognorm_mu=cfg_stage.get('lognorm_mu'),
                lognorm_sigma=cfg_stage.get('lognorm_sigma')
            )
    for stage in DYNAMIC_STAGES:
        cfg_stage = cfg.MACHINE_DEFINITIONS.get(stage, cfg.DEFAULT_MACHINE_CONFIG)
        n = int(cfg_stage.get('num_machines', 1))
        if n > 1:
            stage_machines = [create_machine_from_config(env, stage, cfg.MACHINE_DEFINITIONS, financial_tracker, is_worker=False, index=i)
                              for i in range(n)]
            machines[stage] = stage_machines
            workers[stage] = create_machine_from_config(env, stage, cfg.MACHINE_DEFINITIONS, financial_tracker, is_worker=True)
            print(f"Created {n} parallel {stage} machines")
        else:
            machines[stage] = create_machine_from_config(env, stage, cfg.MACHINE_DEFINITIONS, financial_tracker, is_worker=False)
            workers[stage] = create_machine_from_config(env, stage, cfg.MACHINE_DEFINITIONS, financial_tracker, is_worker=True)
            print(f"Created {stage} machine - MTBF: {cfg.MACHINE_DEFINITIONS.get(stage, cfg.DEFAULT_MACHINE_CONFIG)['mtbf']} hours")
    workers = {}
    metrics = {s: {k: [] for k in ['start_times', 'completion_times', 'stage_flow_times']} for s in DYNAMIC_STAGES}
    job_metrics = {}
    export_kwargs = {
        "job_logger": job_logger,
        "machines": machines,
        "inventory_manager": inventory_manager,
        "order_tracker": order_tracker,
        "financial_tracker": financial_tracker,
        "environmental": environmental,
        "metrics": rt_metrics,
        "qualitytracker": qualitytracker
    }
    print("\nStarting simulation with orders, FG shipping, calendar, extended OEE, inventory, and reliability...")
    snapshot_interval_hours = 1.0  
    order_source = getattr(cfg, "ORDER_SOURCE", "CSV")
    print(f"\nOrder Source configured as: '{order_source}'")
    orders_to_schedule = []
    planning_data = PlanningData()
    if order_source == "DEMAND_TO_CSV":
        cfg.DEMAND_CONFIG['horizon_days'] = int(cfg.SIMULATION_TIME / 24)
        orders_to_schedule, generated_path = generate_csv_from_demand(
            csv_path=cfg.DEMAND_CONFIG['csv_output_path'],
            demand_config=cfg.DEMAND_CONFIG
        )
        print(f"Using {len(orders_to_schedule)} orders generated directly from demand model.")
    elif order_source == "DYNAMIC":
        print("\n--- Generating orders from live DYNAMIC demand model ---")
        demand_cfg = cfg.PLANNING_CONFIG['DEMAND_MODEL']
        forecast_cfg = cfg.PLANNING_CONFIG['FORECAST_MODEL']
        np.random.seed(demand_cfg.get('rng_seed', cfg.RANDOM_SEED))
        demand_gen = DemandGenerator()
        _, planning_data.actual_demand_history = demand_gen.generate_demand(
            horizon=int(cfg.SIMULATION_TIME / 24),
            base_demand=demand_cfg['base_per_day'],
            model=demand_cfg['model']
        )
        forecaster = Forecaster()
        if forecast_cfg['method'] == 'SES':
            planning_data.forecasted_demand_history = forecaster.simple_exponential_smoothing(
                history=planning_data.actual_demand_history,
                alpha=forecast_cfg['alpha']
            )
        product_mix_names = list(demand_cfg['product_mix'].keys())
        product_mix_probs = list(demand_cfg['product_mix'].values())
        order_counter = 0
        for day_idx, daily_total_qty in enumerate(planning_data.actual_demand_history):
            num_orders_today = int(round(daily_total_qty))
            if num_orders_today <= 0: continue
            product_choices = np.random.choice(product_mix_names, size=num_orders_today, p=product_mix_probs)
            for product_name in product_choices:
                order_counter += 1
                orders_to_schedule.append({
                    'id': f'Ord-{order_counter:05d}',
                    'releasetime': day_idx * 24.0,
                    'products': [product_name],
                    'due_in_hours': 7 * 24
                })
        print(f"Dynamically generated {len(orders_to_schedule)} orders.")
    elif order_source == "CSV":
        try:
            orders_to_schedule = load_orders_from_csv(cfg.CSV_ORDERS_PATH)
            print(f"Loading {len(orders_to_schedule)} orders from manual CSV: {cfg.CSV_ORDERS_PATH}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Could not find order file at '{cfg.CSV_ORDERS_PATH}'.")
            return
    elif order_source == "DAILY":
        print("Using legacy DAILY_ORDERS from config is not fully supported with the new scheduler. Please use CSV or DEMAND_TO_CSV.")
        orders_to_schedule = []
    if orders_to_schedule:
        scheduling_mode = getattr(cfg, 'SCHEDULING_MODE', 'EVENT_BASED')
        if scheduling_mode == 'STATIC_PLAN':
            from planning import create_master_sequence
            print("\n--- Generating Static Master Production Sequence ---")
            planning_jobs = [
                {'id': order['id'], 'product_type': order['products'][0]} 
                for order in orders_to_schedule if order.get('products')
            ]
            heuristic_method = getattr(cfg, 'PLANNING_HEURISTIC', 'palmer')
            master_sequence = create_master_sequence(
                jobs=planning_jobs,
                products_config=cfg.PRODUCTS,
                method=heuristic_method
            )
            print(f"Executing with {heuristic_method.upper()} sequence...")
            static_sequence_scheduler(
                env, master_sequence, orders_to_schedule, machines, 
                locals().get('workers'), rt_metrics, job_logger, 
                financial_tracker, inventory_manager, order_tracker, 
                environmental, qualitytracker, calendar, cfg, conwip_container
            )
        else: 
            print("\n--- Executing with EVENT_BASED (Dynamic) Scheduling ---")
            csv_orders_scheduler(
                env, orders_to_schedule, machines, 
                locals().get('workers'), locals().get('metrics'), locals().get('job_metrics'),
                financial_tracker, inventory_manager, order_tracker,
                job_logger, worker_pool, environmental, qualitytracker, calendar, cfg,
                conwip_container=conwip_container
            )
    else:
        print("\nWARNING: No orders were loaded or generated. The simulation will run without any jobs.")
    env.process(daily_shipping_process(env, order_tracker, inventory_manager, qualitytracker, calendar=calendar))
    if hasattr(cfg, 'METRICS_CONFIG') and cfg.METRICS_CONFIG.get('snapshot_enabled', False):
        snapshot_interval = cfg.METRICS_CONFIG.get('snapshot_interval_h', 1.0)
        env.process(snapshot_process(env, snapshot_interval, kpi_calculator, output_dir))
    env.process(daily_shipping_process(env, order_tracker, inventory_manager, qualitytracker, calendar=calendar))
    if hasattr(cfg, 'METRICS_CONFIG') and cfg.METRICS_CONFIG.get('snapshot_enabled', False):
        snapshot_interval = cfg.METRICS_CONFIG.get('snapshot_interval_h', 1.0)
        env.process(snapshot_process(env, snapshot_interval, kpi_calculator, output_dir))
    try:
        worker_pool = WorkerPool(env, calendar, financial_tracker, cfg.WORKER_DEFINITIONS, rt_metrics)  
    except Exception:
        worker_pool = None
    try:
        transport_manager = TransportationManager(
            env,
            worker_pool=worker_pool,
            financial_tracker=financial_tracker,
            inventory_manager=inventory_manager,
            config=cfg.TRANSPORTCONFIG,
            environmental=environmental
        )
    except Exception:
        transport_manager = None
    print("Initializing comprehensive metrics and alert system...")
    kpi_calculator = ComprehensiveKPIs(metrics_bus=rt_metrics,
                                       worker_pool=worker_pool,
    machines=machines,
    financial_tracker=financial_tracker, 
    quality_tracker=qualitytracker,
    job_logger=job_logger, 
    env=env,
    job_metrics=job_metrics, 
    output_dir=output_dir,
    cfg=cfg
    )
    print("\nCalculating total overhead labor cost...")
    calendar = ShiftCalendar(CALENDAR_CONFIG)
    total_working_hours = 0
    simulation_end_time = env.now
    for hour in range(int(math.ceil(simulation_end_time))):
        if calendar.is_working_time(hour):
            total_working_hours += 1
    total_overhead_labor_cost = total_working_hours * WORKER_POOL_CONFIG['total_workers'] * FINANCIAL_CONFIG['labor_cost_per_hour']
    financial_tracker.total_costs += total_overhead_labor_cost
    print(f"Total Working Hours: {total_working_hours}, Workers: {WORKER_POOL_CONFIG['total_workers']}")
    print(f"Total Overhead Labor Cost of ${total_overhead_labor_cost:,.2f} added to total costs.")
    bottleneck_detector = BottleneckDetector(
        machines=machines,
        metrics_bus=rt_metrics,
        config=cfg.METRICS_CONFIG 
    )
    alerts_manager = AlertsManager(
        metrics_bus=rt_metrics,
        financial_tracker=financial_tracker,
        qualitytracker=qualitytracker,
        config=cfg.METRICS_CONFIG
    )
    def metrics_monitor(env):
        yield env.timeout(cfg.METRICS_CONFIG.get('warmup_period', 60))
        while True:
            oee_data = kpi_calculator.update_all(env.now)
            bottleneck_detector.analyze(env.now)
            alerts_manager.evaluate(env.now, oee_data=oee_data)
            yield env.timeout(cfg.METRICS_CONFIG.get('analysis_interval', 10))
    env.process(metrics_monitor(env))
    env.run(until=cfg.SIMULATION_TIME)
    print("="*30 + " DEBUG: RAW DATA CHECK " + "="*30)
    print("="*30 + " DEBUG 1: RAW DATA CHECK " + "="*30)
    any_data_collected = False
    for stage_name, machine_list in machines.items():
        if not isinstance(machine_list, list): machine_list = [machine_list]
        for machine in machine_list:
            if machine.arrival_timestamps or machine.service_times:
                any_data_collected = True
            print(f"  Machine: {machine.name} | Arrivals: {len(machine.arrival_timestamps)} | Services: {len(machine.service_times)}")
    if not any_data_collected:
        print("  [CRITICAL FAILURE] No raw arrival or service data was collected on any machine.")
    print("="*82 + "\n")
    if ANALYTICAL_QUEUE_CONFIG.get("enabled"):
            print("="*50)
            print("PERFORMING POST-SIMULATION ANALYTICAL CALCULATIONS")
            print("="*50)
            for stage_name, machine_list in machines.items():
                if not isinstance(machine_list, list): machine_list = [machine_list]
                for machine_instance in machine_list:
                    print(f"  -> Analyzing machine: {machine_instance.name}")
                    machine_instance.calculate_analytical_metrics()
                    print(f"  Result for {machine_instance.name}: {machine_instance.analytical_results}")
            print("="*50 + "\n")
    print("Calculating and exporting final KPIs...")
    if hasattr(kpi_calculator, 'calculate_and_export'):
        kpi_calculator.calculate_and_export()
        print("KPI export process complete.")
    else:
        print("ERROR: kpi_calculator does not have 'calculate_and_export' method.")
    print("Finalizing all machine state timelines for Gantt chart...")
    for stage_name, machine_or_list in machines.items():
        if isinstance(machine_or_list, list):
            for machine in machine_or_list:
                if hasattr(machine, 'timeline'): machine.timeline.end_all(env.now)
        else:
            if hasattr(machine_or_list, 'timeline'): machine_or_list.timeline.end_all(env.now)
    for proc in list(globals().get("LONG_LIVED_PROCS", [])):
        try:
            proc.interrupt("shutdown")
        except Exception:
            pass
    try:
        env.step()
    except Exception:
        pass
    import matplotlib.pyplot as plt
    try:
        for num in plt.get_fignums():
            try:
                fig = plt.figure(num)
                if hasattr(fig.canvas.manager, 'key_press_handler_id'):
                    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            except Exception:
                pass
        plt.close('all')
    except Exception:
        pass
    print(f"Simulation completed at time {env.now:.2f}")
    print("\n" + "="*40 + "\nTIME-WEIGHTED AVERAGE WORKER UTILIZATION\n" + "="*40 + "\n")
    try:
        util_series = None
        if rt_metrics and hasattr(rt_metrics, 'series'):
            util_series = rt_metrics.series.get('worker_utilization')
        if util_series and len(util_series) > 1:
            total_duration = 0
            weighted_util_sum = 0
            for i in range(len(util_series) - 1):
                timestamp1, util1 = util_series[i]
                timestamp2, util2 = util_series[i+1]
                duration = timestamp2 - timestamp1
                if duration < 0: continue
                weighted_util_sum += util1 * duration
                total_duration += duration
            last_timestamp, last_util = util_series[-1]
            final_duration = cfg.SIMULATION_TIME - last_timestamp
            if final_duration > 0:
                weighted_util_sum += last_util * final_duration
                total_duration += final_duration
            average_utilization = weighted_util_sum / total_duration if total_duration > 0 else 0
            print(f"Average Worker Utilization over {total_duration:.2f} hours: {average_utilization:.2%}")
        elif util_series:
            last_util = util_series[0][1]
            print(f"Average Worker Utilization: {last_util:.2%}")
        else:
            print("No worker utilization data was recorded in 'rt_metrics'.")
    except Exception as e:
        print(f"Could not calculate average worker utilization. Error: {e}")
    print("\n" + "="*25 + "\nFACTORY PERFORMANCE REPORT\n" + "="*25)
    total_jobs = len(job_metrics)
    print(f"Total Jobs Completed: {total_jobs}")
    print("\n" + "="*30 + "\nBREAKDOWN & MAINTENANCE REPORT\n" + "="*30)
    total_downtime = 0.0
    for stage_name, m in machines.items():
        machine_list = m if isinstance(m, list) else [m]
        stage_has_multiple = len(machine_list) > 1
        if stage_has_multiple:
            print(f"{stage_name} (parallel machines):")
        for mi in machine_list:
            mi.timeline.end_all(env.now)
            availability_simple = ((env.now - mi.total_downtime) / env.now) * 100 if env.now > 0 else 0.0
            total_downtime += mi.total_downtime
            label = getattr(mi, "name", f"{stage_name}_Machine")
            indent = "  " if stage_has_multiple else ""
            print(f"{indent}{label}:")
            print(f"{indent}  Breakdowns: {mi.breakdown_count}")
            print(f"{indent}  Maintenance cycles: {mi.maintenance_count}")
            print(f"{indent}  Total downtime: {mi.total_downtime:.2f} hours")
            print(f"{indent}  Simple Availability: {availability_simple:.1f}%")
            print(f"{indent}  MTBF (current): {mi.mtbf:.1f} hours")
    num_physical = sum(len(m) if isinstance(m, list) else 1 for m in machines.values())
    overall_availability = (((env.now * num_physical) - total_downtime) / (env.now * num_physical)) * 100 if env.now > 0 and num_physical > 0 else 0.0
    print(f"\nOverall Factory Availability (simple): {overall_availability:.1f}%")
    print("\n" + "="*40 + "\nFINANCIAL PERFORMANCE REPORT\n" + "="*40)
    financial_summary_full = financial_tracker.get_financial_summary(cfg.SIMULATION_TIME)
    print(f"Simulation Period: {cfg.SIMULATION_TIME:.1f} hours")
    print(f"Total Revenue: ${financial_summary_full['total_revenue']:,.2f}")
    print(f"Total Costs: ${financial_summary_full['total_costs']:,.2f}")
    print(f"  - Downtime Costs: ${financial_summary_full['total_downtime_cost']:,.2f}")
    print(f"Net Profit: ${financial_summary_full['net_profit']:,.2f}")
    print(f"Profit Margin: {financial_summary_full['profit_margin']:.1f}%")
    print(f"Revenue per Hour: ${financial_summary_full['revenue_per_hour']:,.2f}")
    print(f"Profit per Hour: ${financial_summary_full['profit_per_hour']:,.2f}")
    print(f"\n--- PRODUCT PERFORMANCE ---")
    for product_type, perf in financial_summary_full['product_performance'].items():
        print(f"{product_type}:")
        print(f"  Units Produced: {perf['units']}")
        print(f"  Revenue: ${perf['revenue']:,.2f}")
        print(f"  Net Profit: ${perf['net_profit']:,.2f}")
        print(f"  Profit per Unit: ${perf['profit_per_unit']:,.2f}")
        print(f"  Avg Cycle Time: {perf['avg_cycle_time']:.1f} hours")
    print(f"\n--- DOWNTIME COST BREAKDOWN ---")
    breakdown_cost = sum(d['total_cost'] for d in financial_tracker.downtime_losses if d['event_type'] == 'breakdown')
    maintenance_cost = sum(d['total_cost'] for d in financial_tracker.downtime_losses if d['event_type'] == 'maintenance')
    print(f"Breakdown Costs: ${breakdown_cost:,.2f}")
    print(f"Maintenance Costs: ${maintenance_cost:,.2f}")
    if financial_summary_full['total_revenue'] > 0:
        print(f"Downtime Impact: {(financial_summary_full['total_downtime_cost']/financial_summary_full['total_revenue']*100):.1f}% of revenue")
    print("\n" + "="*40 + "\nINVENTORY MANAGEMENT REPORT\n" + "="*40)
    inventory_status = inventory_manager.get_inventory_status()
    inventory_metrics = inventory_manager.get_inventory_metrics()
    print("Current Stock Levels:")
    for material, status in inventory_status.items():
        print(f"  {material}: {status['current_stock']:.1f} units ({status['status']}) - {status['stock_ratio']*100:.1f}% of target")
    print(f"\nInventory Performance:")
    print(f"  Total Procurement Cost: ${inventory_metrics['total_procurement_cost']:,.2f}")
    print(f"  Total Stockouts: {inventory_metrics['total_stockouts']}")
    print(f"  Orders Placed: {inventory_metrics['total_orders_placed']}")
    print("\n" + "="*40 + "\nPENDING JOBS & QUEUE ANALYSIS\n" + "="*40)
    current_queues = {}
    for stage, machine_or_list in machines.items():
        if not isinstance(machine_or_list, list):
            machine_or_list = [machine_or_list]
        total_queue_for_stage = 0
        for m in machine_or_list:
            if hasattr(m, 'queue') and hasattr(m.queue, 'jobs'):
                total_queue_for_stage += len(m.queue.jobs)
        current_queues[stage] = total_queue_for_stage
    print("Final Queue Status at End of Simulation:")
    if any(q > 0 for q in current_queues.values()):
        for stage, count in current_queues.items():
            if count > 0:
                print(f"  {stage}: {count} jobs pending")
    else:
        print("  All machine queues are empty.")
    print("\nNOTE: Detailed queue performance statistics (wait times, etc.)")
    print("are now logged continuously to 'queue_events.csv' in your output folder for detailed analysis.")
    print("\n" + "="*30 + "\nOEE REPORT\n" + "="*30)
    overall_oee_weighted = 0.0
    overall_ppt_sum = 0.0
    for stage_name, m in machines.items():
        machine_list = m if isinstance(m, list) else [m]
        stage_ppt_sum = 0.0
        stage_oee_weighted = 0.0
        stage_header_printed = False
        for mi in machine_list:
            mi.timeline.end_all(env.now)
            oee = compute_oee_from_timeline(
                mi.timeline, env.now,
                availability_counts_pm=True,
                good_units=None, total_units=None,
                observed_run_time=None, ideal_run_time=None
            )
            try:
                if isinstance(oee, dict):
                    rt_metrics.observe("oee", float(oee.get('oee', 0.0)), env.now if hasattr(env, "now") else cfg.SIMULATION_TIME)
            except Exception:
                pass
            if len(machine_list) > 1 and not stage_header_printed:
                print(f"{stage_name} (parallel machines):")
                stage_header_printed = True
            label = getattr(mi, "name", f"{stage_name}_Machine")
            indent = "  " if len(machine_list) > 1 else ""
            print(f"{indent}{label}:")
            print(f"{indent}  PPT: {oee['ppt']:.2f}h | Calendar Off: {oee['calendar_off']:.2f}h")
            print(f"{indent}  Unplanned DT: {oee['unplanned_dt']:.2f}h | PM DT: {oee['pm_dt']:.2f}h")
            print(f"{indent}  Running: {oee['running_time']:.2f}h | Setup: {oee['setup_time']:.2f}h | Wait: {oee['wait_time']:.2f}h")
            print(f"{indent}  Availability: {oee['availability']*100:.1f}% | Performance: {oee['performance']*100:.1f}% | Quality: {oee['quality']*100:.1f}%")
            print(f"{indent}  OEE: {oee['oee']*100:.1f}%")
        if stage_ppt_sum > 0 and len(machine_list) > 1:
            stage_oee = stage_oee_weighted / stage_ppt_sum
            print(f"  {stage_name} Aggregate OEE (PPT-weighted): {stage_oee * 100:.1f}%")
        overall_oee_weighted += stage_oee_weighted
        overall_ppt_sum += stage_ppt_sum
    if overall_ppt_sum > 0:
        print(f"\nOverall OEE (PPT-weighted): {overall_oee_weighted / overall_ppt_sum * 100:.1f}%")
    print("\n" + "="*40 + "\nORDER TRACKING REPORT\n" + "="*40)
    summary = order_tracker.summary()
    print(f"Open orders: {summary.get('open', [])}")
    print(f"In-progress orders: {summary.get('in_progress', [])}")
    print(f"Completed (awaiting shipment): {summary.get('completed', [])}")
    print(f"Shipped orders: {summary.get('shipped', [])}")
    for oid, od in order_tracker.orders.items():
        done, total = order_tracker.order_progress(oid)
        print(f"\nOrder {oid}: {done}/{total} items completed")
        print(f"  Released: {od['release_time']:.2f}")
        if od['start_time'] is not None:
            print(f"  Started: {od['start_time']:.2f}")
        if od['completion_time'] is not None:
            lead = od['completion_time'] - od['release_time']
            print(f"  Completed: {od['completion_time']:.2f} (Lead: {lead:.2f}h)")
        if od['due_time'] is not None and od['completion_time'] is not None:
            late = order_tracker.lateness(oid)
            if late is not None:
                print(f"  Lateness: {late:.2f}h")
        if od['shipped_time'] is not None:
            print(f"  Shipped: {od['shipped_time']:.2f}")
    print("\n" + "="*40 + "\nFINISHED GOODS (TO BE SHIPPED)\n" + "="*40)
    if inventory_manager.finished_goods:
        for p, q in inventory_manager.finished_goods.items():
            print(f"  {p}: {q} units")
    else:
        print("  None")
    print("\n" + "="*40 + "\nSHIPPED SUMMARY\n" + "="*40)
    if inventory_manager.shipped:
        for p, q in inventory_manager.shipped.items():
            print(f"  {p}: {q} units shipped")
    else:
        print("  None")
    if 'worker_pool' in locals() and worker_pool is not None:
        print("=== WORKER POOL DEBUG ===")
        utilization = worker_pool.get_worker_utilization()
        print(f"Total workers: {utilization['total_workers']}")
        print(f"Available workers: {utilization['available']}")
        print(f"Working workers: {utilization['working']}")
        print("\n=== SKILL MATRIX DEBUG ===")
        for stage in ['Cutting', 'Routing', 'Painting', 'Assembling']:
            if hasattr(worker_pool, 'skill_matrix') and stage in worker_pool.skill_matrix:
                total_qualified = len(worker_pool.skill_matrix[stage])
                currently_available = sum(1 for w in worker_pool.skill_matrix[stage] if w.is_currently_available())
                print(f"{stage}: {currently_available}/{total_qualified} workers available")
            else:
                print(f"{stage}: No workers qualified")
        print(f"\nPending worker requests: {len(getattr(worker_pool, 'waiting_requests', []))}")
    if output_dir is None:
        output_dir = "./out/baseline"
    _feed_environment_from_timelines(machines, environmental)
    financial_summary = {
        "simulationtime": float(getattr(cfg, "SIMULATION_TIME", 0.0)),
        "totalrevenue": float(financial_summary_full['total_revenue']),
        "totalcosts": float(financial_summary_full['total_costs']),
        "netprofit": float(financial_summary_full['net_profit'])
    }
    from logging_export import export_all, export_transport_costs
    if callable(job_logger):
        raise TypeError("job_logger is callable; expected an instance with .events")
    for _name in ("inventory_manager", "order_tracker", "financial_tracker", "environmental", "qualitytracker"):
        if callable(locals().get(_name)):
            raise TypeError(f"{_name} is callable; expected an instance")
    if not isinstance(machines, dict):
        raise TypeError("machines must be a dict of stage -> machine/list")
    if callable(rt_metrics):
        rt_metrics = None  
    if callable(job_metrics):
        job_metrics = {}   
    try:
        if worker_pool is not None and rt_metrics is not None:
            current_util = ensure_worker_utilization_tracking(worker_pool, rt_metrics, env.now)
            print(f"Quality events count: {len(qualitytracker.quality_events)}")
            print(f"Pre-export worker utilization: {current_util:.3f}")
        calendar = ShiftCalendar(cfg.CALENDAR_CONFIG)
        total_working_hours = 0
        simulation_end_time = env.now
        for hour in range(int(math.ceil(simulation_end_time))):
            if calendar.is_working_time(hour):
                total_working_hours += 1
        total_overhead_labor_cost = total_working_hours * cfg.WORKER_POOL_CONFIG['total_workers'] * cfg.FINANCIAL_CONFIG['labor_cost_per_hour']
        financial_tracker.total_costs += total_overhead_labor_cost
        print(f"Total Overhead Labor Cost of {total_overhead_labor_cost:,.2f} added to total costs.")
        try:
            print("\n[INFO] Running advanced financial analysis...")
            fin_analysis = FinancialAnalysis(
                financial_tracker=financial_tracker,
                quality_tracker=qualitytracker,
                inventory_manager=inventory_manager,
                machines=machines,
                worker_pool=worker_pool,
                env_tracker=environmental,
                config=cfg,
                simulation_time_hours=env.now,
                total_labor_cost=total_overhead_labor_cost
            )
            advanced_metrics = fin_analysis.calculate_all_metrics()
            financial_viz_data = fin_analysis.generate_visualization_data()
            export_kwargs['advanced_financials'] = advanced_metrics
            export_kwargs['financial_viz_data'] = financial_viz_data
            print("[INFO] Advanced financial analysis complete.")
        except Exception as e:
            print(f"\n[ERROR] Could not run advanced financial analysis: {e}")
            import traceback
            traceback.print_exc()
            export_kwargs['advanced_financials'] = {}
            export_kwargs['financial_viz_data'] = {}
        print("\n" + "="*30)
        print("FINAL PRE-EXPORT OBJECT INSPECTION")
        print(f"kpi_calculator object ID: {id(kpi_calculator)}")
        if hasattr(kpi_calculator, 'analytical_comparison'):
            print("Attribute 'analytical_comparison' EXISTS.")
            print(f"Content: {kpi_calculator.analytical_comparison}")
        else:
            print("CRITICAL: Attribute 'analytical_comparison' DOES NOT EXIST on this object.")
        print("="*30 + "\n")
        export_all(
            output_dir,
            job_logger,            
            machines,              
            inventory_manager,     
            order_tracker,         
            financial_summary,     
            financial_tracker,
            worker_pool,
            environmental,
            metrics=rt_metrics,    
            job_metrics=job_metrics,       
            qualitytracker=qualitytracker,
            kpi_calculator=kpi_calculator,
            config_ns=cfg,
            advanced_financials=advanced_metrics,
            financial_viz_data=financial_viz_data)
    except Exception as e:
        print(f"Export failed with error: {e}")
    try:
        export_transport_costs(financial_tracker, output_dir)
    except Exception as e:
        print(f"Transport cost export failed with error: {e}")
    print(f"Exports written to {output_dir}")
    return financial_summary
if __name__ == '__main__':
    
    import config as baseconfig

    print("--- EXECUTING DIRECT RUN FROM MAIN.PY ---")
    
    # OUTPUT DIR
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    outputdir = os.path.join(base_dir, "data", "processed", f"direct_run-{int(time.time())}")
    os.makedirs(outputdir, exist_ok=True)
    scenariolabel = "DirectRun"
    
    print(f"Using default config. Outputs will be saved to: {outputdir}")

    try:
        summary = run_simulation(
            config_ns=baseconfig, 
            output_dir=outputdir, 
            scenario_label=scenariolabel
        )
        print("--- Direct Run Complete ---")
        if summary:
            print("KPI Summary:", summary)
            
    except Exception as e:
        print("--- DIRECT RUN FAILED ---")
        print(f"An error occurred: {e}")
