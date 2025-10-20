import math
from dispatch_policies import DispatchPolicyBase, get_dispatch_policy
USE_CSV_ORDERS = True
CSV_ORDERS_PATH = "data/raw/orders.csv"
SIMULATION_TIME = 200 
RANDOM_SEED = 42       
WARMUP_PERIOD = 0
SCHEDULING_MODE = 'EVENT_BASED' 
PLANNING_HEURISTIC = 'cds' 
ORDER_SOURCE = "DYNAMIC" 
DEMAND_CONFIG = {
    "horizon_days": 90,
    "base_per_day": 10,           
    "model": "seasonal_noise",
    "rng_seed": 42,
    "csv_output_path": "data/raw/orders/generated_demand_orders.csv",
    "product_mix": {
        "ProductA": 0.4,                
        "ProductB": 0.3,                
        "ProductD_with_Assembly": 0.2,  
        "ProductC": 0.1         
    },
    "due_lag_days": 7,
    "per_unit_orders": True
}
FINANCIAL_CONFIG = {
    'labor_cost_per_hour': 20,      
    'downtime_cost_per_hour': 40,
    'stockout_cost_per_unit': 15.0,
}
CALENDAR_CONFIG = {
    'week_length': 7 * 24,             
    'working_days': [0, 1, 2, 3, 4, 5],
    'daily_shifts': [
        {'start': 6,  'end': 14, 'breaks': [(10, 10.5)]},  
        {'start': 14, 'end': 22, 'breaks': [(18, 18.5)]},  
    ],
    'holidays': [
    ],
    'machine_overrides': {
    },
    'maintenance_windows': {
    }
}
CALENDAR_CONFIG["maintenance_windows"] = CALENDAR_CONFIG.get("maintenance_windows", {})
CALENDAR_CONFIG["maintenance_windows"]["Painting_Machine1"] = {"every_hours": 500, "offset": 0}       
CALENDAR_CONFIG["maintenance_windows"]["Painting_Machine2"] = {"every_hours": 500, "offset": 125}     
CALENDAR_CONFIG["maintenance_windows"]["Painting_Machine3"] = {"every_hours": 500, "offset": 250}     
CALENDAR_CONFIG["machine_overrides"] = CALENDAR_CONFIG.get("machine_overrides", {})
CALENDAR_CONFIG["machine_overrides"]["Painting"] = {
    "daily_shifts": CALENDAR_CONFIG["daily_shifts"],   
    "working_days": CALENDAR_CONFIG["working_days"]    
}
SHIPPING_CONFIG = {
    'daily_shipping_hour': 19.0,   
    'ship_partial_orders': True    
}
ENVIRONMENTCONFIG = {
  "enabled": True,
  "grid_emission_factor_kg_per_kwh": 0.7,
  "diesel_emission_factor_kg_per_l": 2.68,
  "water_emission_factor_kg_per_m3": 0.35,
  "kwh_per_l_diesel_equiv": 9.7,
  "report_states": ["running", "setup", "idle", "waitingworker", "waitingmaterial", "maintenance", "breakdown", "calendaroff"],
  "enable_water_tracking": True,
  "enable_waste_tracking": True
}
QUALITYCONFIG = {
    "enabled": True,
    "defect_p": {
        "default": {"Cutting": 0.01, "Routing": 0.015, "Painting": 0.03, "Assembling": 0.025},
        "ProductA": {"Painting": 0.025},
        "ProductB": {},
        "ProductC": {},
        "ProductD_with_Assembly": {"Assembling": 0.03}
    },
    "detect_p": {"with_qc": 0.9, "no_qc": 0.6},
    "rework": {"enabled": True, "max_loops": 1, "time_factor": 0.5, "success_p": 0.7},
    "scrap_cost_per_unit": {"default": 10.0, "ProductA": 12.0, "ProductD_with_Assembly": 18.0},
    "returns": {"enabled": True, "escape_to_return_p": 0.5, "delay_h": 24.0, "refund_fraction": 1.0}
}
FLOW_CONTROL_CONFIG = {
    'enabled': True,  
    'wip_limit': 20
}
SETUP_MATRIX = {
    'ProductA': {
        'ProductB': 1.5,
        'ProductC': 1.2,
        'ProductDwithAssembly': 2.0,
    },
    'ProductB': {
        'ProductA': 1.8,
        'ProductC': 1.0,
        'ProductDwithAssembly': 2.2,
    },
    'ProductC': {
        'ProductA': 1.1,
        'ProductB': 0.9,
        'ProductDwithAssembly': 1.5,
    },
    'ProductDwithAssembly': {
        'ProductA': 2.5,
        'ProductB': 2.3,
        'ProductC': 1.7,
    }
}
PRODUCTS = {
    'ProductA': {
        'routing': ['Cutting', 'Routing', 'Painting'],
        'machine_times': {'Cutting': (3, 4), 'Routing': (4, 5), 'Painting': (5, 6)},
        'load_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'unload_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'setup_times': {'Cutting': 1, 'Routing': 2, 'Painting': 1},
        'profit_per_unit': 100,
        'color': 'skyblue',
        'unit_price': 300,
        'unit_cogs': 180
    },
    'ProductB': {
        'routing': ['Routing', 'Cutting', 'Painting'],
        'machine_times': {'Cutting': (2, 3), 'Routing': (3, 6), 'Painting': (4, 5)},
        'load_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'unload_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'setup_times': {'Cutting': 2, 'Routing': 1, 'Painting': 2},
        'profit_per_unit': 120,
        'color': 'salmon',
        'unit_price': 300,
        'unit_cogs': 180
    },
    'ProductC': {
        'routing': ['Cutting', 'Painting'],
        'machine_times': {'Cutting': (4, 5), 'Painting': (6, 7)},
        'load_times': {'Cutting': (1, 1), 'Painting': (2, 2)},
        'unload_times': {'Cutting': (1, 1), 'Painting': (2, 2)},
        'setup_times': {'Cutting': 1.5, 'Painting': 1.5},
        'profit_per_unit': 90,
        'color': 'lightgreen',
        'unit_price': 300,
        'unit_cogs': 180
    },
    'ProductD_with_Assembly': {
        'routing': ['Cutting', 'Assembling', 'Painting'],
        'machine_times': {'Cutting': (2, 3), 'Assembling': (5, 8), 'Painting': (4, 5)},
        'load_times': {'Cutting': (1, 2), 'Assembling': (1, 1), 'Painting': (1, 2)},
        'unload_times': {'Cutting': (1, 2), 'Assembling': (1, 1), 'Painting': (1, 2)},
        'setup_times': {'Cutting': 1, 'Assembling': 2, 'Painting': 1.5},
        'profit_per_unit': 150,
        'color': 'gold',
        'unit_price': 300,
        'unit_cogs': 180
    }
}
MACHINE_DEFINITIONS = {
'Cutting': {
    'capacity': 1,
    'mtbf': 300,
    'mttr': 4,
    'maintenance_interval': 600,
    'operating_cost_per_hour': 5,
    'maintenance_cost': 120,
    'breakdown_repair_multiplier': 2.0,
    'color': 'red',
    'num_machines': 4,
    'failure_model': 'exponential',
    "analytical_model": {
            "type": "M/M/1", 
            "arrival_rate_lambda": 0.2, 
            "service_rate_mu": 0.25,    
            "arrival_scv": 1.0,
            "service_scv": 1.0
    }},
'Routing': {
    'capacity': 1,
    'mtbf': 400,
    'mttr': 6,
    'maintenance_interval': 700,
    'operating_cost_per_hour': 8,
    'maintenance_cost': 150,
    'breakdown_repair_multiplier': 1.8,
    'color': 'blue',
    'num_machines': 4,
    'failure_model': 'weibull',
    'weibull_k': 2.0,
    'weibull_lambda': 450,
    "analytical_model": {
            "type": "M/G/1", 
            "arrival_rate_lambda": 0.18,
            "service_rate_mu": 0.22,
            "arrival_scv": 1.0,
            "service_scv": 1.0
    }},
'Painting': {
    'capacity': 1,
    'mtbf': 250,
    'mttr': 5,
    'maintenance_interval': 500,
    'operating_cost_per_hour': 2,
    'maintenance_cost': 140,
    'breakdown_repair_multiplier': 2.2,
    'color': 'green',
    'num_machines': 4,
    'failure_model': 'lognormal',
    'lognorm_mu': 5.5,    
    'lognorm_sigma': 0.4,
    "analytical_model": {
            "type": "M/M/c", 
            "arrival_rate_lambda": 0.5,
            "service_rate_mu": 0.2, 
            "arrival_scv": 1.0,
            "service_scv": 1.0  
    }},
'Assembling': {
    'capacity': 1,
    'mtbf': 150,
    'mttr': 8,
    'maintenance_interval': 400,
    'operating_cost_per_hour': 5,
    'maintenance_cost': 200,
    'breakdown_repair_multiplier': 1.5,
    'color': 'purple',
    'num_machines': 4,
    'failure_model': 'weibull',
    'weibull_k': 1.5,
    'weibull_lambda': 180,
    "analytical_model": {
            "type": "M/G/c", 
            "type": "M/M/c", 
            "arrival_rate_lambda": 0.3,
            "service_rate_mu": 0.15, 
            "arrival_scv": 1.0,
            "service_scv": 1.0
}
}}
MACHINE_DEFINITIONS["Painting"].update({
    "mtbf": 320,              
    "mttr": 5,                
    "lognorm_sigma": 0.25      
})
DEFAULT_MACHINE_CONFIG = {
    'capacity': 1,
    'mtbf': 300,
    'mttr': 6,
    'maintenance_interval': 600,
    'operating_cost_per_hour': 5,
    'maintenance_cost': 150,
    'breakdown_repair_multiplier': 1.5,
    'color': 'gray',
    'num_machines': 2,
    'failure_model': 'exponential'
}
MACHINE_ENERGY_PROFILES = {
  "Cutting": {
    "running_kW": 15.0,
    "setup_kW": 10.0,
    "idle_kW": 4.0,
    "waitingworker_kW": 3.0,
    "waitingmaterial_kW": 3.0,
    "maintenance_kW": 2.5,
    "breakdown_kW": 2.0,
    "calendaroff_kW": 0.05
  },
  "Routing": {
    "running_kW": 12.0,
    "setup_kW": 8.0,
    "idle_kW": 3.5,
    "waitingworker_kW": 2.5,
    "waitingmaterial_kW": 2.5,
    "maintenance_kW": 2.0,
    "breakdown_kW": 1.5,
    "calendaroff_kW": 0.05
  },
  "Painting": {
    "running_kW": 20.0,
    "setup_kW": 12.0,
    "idle_kW": 6.0,
    "waitingworker_kW": 4.0,
    "waitingmaterial_kW": 4.0,
    "maintenance_kW": 3.0,
    "breakdown_kW": 2.0,
    "calendaroff_kW": 0.08
  },
  "Assembling": {
    "running_kW": 10.0,
    "setup_kW": 6.0,
    "idle_kW": 2.5,
    "waitingworker_kW": 2.0,
    "waitingmaterial_kW": 2.0,
    "maintenance_kW": 1.5,
    "breakdown_kW": 1.0,
    "calendaroff_kW": 0.04
  },
  "TransportDock": {
    "running_kW": 4.0,
    "setup_kW": 2.0,
    "idle_kW": 1.0,
    "waitingworker_kW": 1.0,
    "waitingmaterial_kW": 1.0,
    "maintenance_kW": 0.8,
    "breakdown_kW": 0.5,
    "calendaroff_kW": 0.02
  }
}
TRANSPORT_ENV = {
  "forklift": {
    "energy_mode": "diesel",
    "l_per_km": 0.08,
    "kwh_per_km": 0.0
  },
  "agv": {
    "energy_mode": "electric",
    "l_per_km": 0.0,
    "kwh_per_km": 0.5
  },
  "conveyor": {
    "energy_mode": "electric",
    "kwh_per_km": 1.0,
    "l_per_km": 0.0
  }
}
ANALYTICAL_QUEUE_CONFIG = {
    "enabled": True,
    "jackson_network_enabled": True,
    "parameter_source": "auto" 
}
DAILY_ORDERS = [
    {'order_id': 'Day1', 'products': ['ProductA', 'ProductA', 'ProductB', 'ProductC'], 'next_order_arrival_delay': 8, 'due_in_hours': 48},
    {'order_id': 'Day2', 'products': ['ProductB', 'ProductB', 'ProductA', 'ProductD_with_Assembly'], 'next_order_arrival_delay': 8, 'due_in_hours': 48},
    {'order_id': 'Day3', 'products': ['ProductC', 'ProductC', 'ProductA', 'ProductB', 'ProductD_with_Assembly'], 'next_order_arrival_delay': 0, 'due_in_hours': 48}
]
CHART_CONFIG = {
    'figure_size': (18, 10),
    'show_annotations': True,
    'annotation_threshold': 3,
    'gantt_title': 'Factory Production & Maintenance Schedule',
    'breakdown_color': 'darkred',
    'maintenance_color': 'orange'
}
PRODUCT_ENV = {
  "ProductA": {"waste_kg": 0.2, "recycle_rate": 0.5, "water_m3": 0.05},
  "ProductB": {"waste_kg": 0.25, "recycle_rate": 0.45, "water_m3": 0.06},
  "ProductC": {"waste_kg": 0.3, "recycle_rate": 0.4, "water_m3": 0.04},
  "ProductD_with_Assembly": {"waste_kg": 0.35, "recycle_rate": 0.5, "water_m3": 0.08}
}
MATERIAL_ENV = {
  "Steel_Sheet": {"waste_factor": 0.05, "water_m3_per_unit": 0.001},
  "Paint": {"waste_factor": 0.03, "water_m3_per_unit": 0.0},
  "Routing_Bits": {"waste_factor": 0.02, "water_m3_per_unit": 0.0},
  "Assembly_Parts": {"waste_factor": 0.01, "water_m3_per_unit": 0.0005}
}
RAW_MATERIALS = {
    'Steel_Sheet': {
        'initial_stock': 150,
        'cost_per_unit': 5.0,
        'used_by_stages': ['Cutting']
    },
    'Paint': {
        'initial_stock': 150,
        'cost_per_unit': 3.0,
        'used_by_stages': ['Painting']
    },
    'Routing_Bits': {
        'initial_stock': 150,
        'cost_per_unit': 2.0,
        'used_by_stages': ['Routing']
    },
    'Assembly_Parts': {
        'initial_stock': 150,
        'cost_per_unit': 8.0,
        'used_by_stages': ['Assembling']
    }
}
INVENTORY_POLICY = {
    'storage_capacity': 1000,
    'Steel_Sheet': {
        'policy': 's_S',
        's': 75,
        'S': 200,
        'available_suppliers': ['Supplier_NA_Reliable', 'Supplier_EU_Standard'] 
    },
    'Paint': {
        'policy': 'BASE_STOCK',
        'review_period_h': 24,
        'target_level': 100,
        'available_suppliers': ['Supplier_EU_Standard', 'Supplier_Asia_Unstable'] 
    },
    'Routing_Bits': {
        'policy': 'REORDER_POINT',
        'reorder_level': 40,
        'order_qty': 100,
        'available_suppliers': ['Supplier_NA_Reliable'] 
    },
    'Assembly_Parts': {
        'policy': 'EOQ',
        'params_eoq': { 'safety_stock_days': 5 }, 
        'available_suppliers': ['Supplier_NA_Reliable']
    }
}
SUPPLY_CHAIN_CONFIG = {
    'enabled': True,
    'risk_model': 'dynamic_events', 
    'procurement_policy': 'best_reliability', 
    'suppliers': {
        'Supplier_NA_Reliable': {
            'region': 'North America',
            'reliability': 0.995,
            'lead_time_dist': {'type': 'normal', 'mean': 24, 'stddev': 4},
            'cost_multiplier': 1.2,
            'min_order_qty': 50
        },
        'Supplier_EU_Standard': {
            'region': 'Europe',
            'reliability': 0.95,
            'lead_time_dist': {'type': 'triangular', 'low': 60, 'mode': 72, 'high': 96},
            'cost_multiplier': 1.0,
            'min_order_qty': 100
        },
        'Supplier_Asia_Unstable': {
            'region': 'Asia',
            'reliability': 0.85,
            'lead_time_dist': {'type': 'lognormal', 'mean': math.log(120), 'stddev': 0.5},
            'cost_multiplier': 0.75,
            'min_order_qty': 200
        }
    },
    'disruption_events': [
        {
            'name': 'Major Port Strike',
            'target_region': 'Asia',
            'start_time': 500,
            'duration': 20,
            'impact': { 'reliability_multiplier': 0.01, 'lead_time_multiplier': 3.0 }
        },
        {
            'name': 'Geopolitical Tensions',
            'target_region': 'Europe',
            'start_time': 400,
            'duration': 300,
            'impact': { 'reliability_multiplier': 0.5, 'lead_time_multiplier': 1.5 }
        }
    ]
}
MATERIAL_CONSUMPTION = {
    'ProductA': {
        'Cutting': {'Steel_Sheet': 2},
        'Routing': {'Routing_Bits': 0.1},
        'Painting': {'Paint': 1}
    },
    'ProductB': {
        'Cutting': {'Steel_Sheet': 1.5},
        'Routing': {'Routing_Bits': 0.15},
        'Painting': {'Paint': 1.2}
    },
    'ProductC': {
        'Cutting': {'Steel_Sheet': 3},
        'Painting': {'Paint': 1.5}
    },
    'ProductD_with_Assembly': {
        'Cutting': {'Steel_Sheet': 2.5},
        'Assembling': {'Assembly_Parts': 1},
        'Painting': {'Paint': 1.8}
    }
}
WORKER_POOL_CONFIG = {
    'total_workers': 15,  
    'shift_overlap_minutes': 30,  
    'fatigue_enabled': True,
    'fatigue_recovery_rate': 0.2,  
    'max_fatigue_threshold': 0.95,  
    'unpredictability_enabled': False,  
}
WORKER_SKILLS = {
    'operator': {
        'description': 'Basic machine operation',
        'compatible_machines': ['Cutting', 'Routing', 'Painting', 'Assembling'],
        'efficiency_multiplier': 1.0,
        'training_time': 40
    },
    'maintenance': {
        'description': 'Machine maintenance and repair',
        'compatible_machines': ['Cutting', 'Routing', 'Painting', 'Assembling'],
        'efficiency_multiplier': 1.2,
        'training_time': 80
    },
    'quality_control': {
        'description': 'Quality inspection and control',
        'compatible_machines': ['Painting', 'Assembling'],
        'efficiency_multiplier': 0.9,
        'training_time': 60
    },
    'specialist_cutting': {
        'description': 'Advanced cutting operations',
        'compatible_machines': ['Cutting'],
        'efficiency_multiplier': 1.3,
        'training_time': 100
    },
    'specialist_painting': {
        'description': 'Advanced painting techniques',
        'compatible_machines': ['Painting'],
        'efficiency_multiplier': 1.25,
        'training_time': 90
    }
}
WORKER_SCHEDULING = {
    'algorithm': 'skill_priority',  
    'priority_weights': {
        'skill_match': 0.35,
        'efficiency': 0.4,
        'fatigue_level': 0.15,
        'experience': 0.1
    },
    'max_consecutive_hours': 12,  
    'mandatory_break_hours': 1,   
}
WORKER_UNPREDICTABILITY = {
    'sick_day_probability': 0.02,     
    'vacation_probability': 0.001,   
    'sick_duration_dist': {'type': 'discrete', 'values': [1, 2, 3, 5], 'probs': [0.5, 0.3, 0.15, 0.05]},
    'vacation_duration_dist': {'type': 'discrete', 'values': [5, 10, 15], 'probs': [0.6, 0.3, 0.1]},
    'late_arrival_probability': 0.0001,  
    'late_arrival_delay_dist': {'type': 'uniform', 'low': 0.25, 'high': 2.0}
}
WORKER_DEFINITIONS = [
    {'id': 'W001', 'name': 'Alice_Senior', 'skills': ['operator', 'maintenance', 'specialist_cutting'], 
     'experience_years': 8, 'base_efficiency': 1.1, 'preferred_shift': 0},
    {'id': 'W002', 'name': 'Bob_Senior', 'skills': ['operator', 'quality_control', 'specialist_painting'], 
     'experience_years': 6, 'base_efficiency': 1.05, 'preferred_shift': 0},
    {'id': 'W003', 'name': 'Carol_Op', 'skills': ['operator'], 
     'experience_years': 3, 'base_efficiency': 1.0, 'preferred_shift': 0},
    {'id': 'W004', 'name': 'David_Op', 'skills': ['operator'], 
     'experience_years': 2, 'base_efficiency': 0.95, 'preferred_shift': 1},
    {'id': 'W005', 'name': 'Eve_Op', 'skills': ['operator', 'quality_control'], 
     'experience_years': 4, 'base_efficiency': 1.02, 'preferred_shift': 1},
    {'id': 'W006', 'name': 'Frank_Op', 'skills': ['operator'], 
     'experience_years': 1, 'base_efficiency': 0.9, 'preferred_shift': 1},
    {'id': 'W007', 'name': 'Grace_Maint', 'skills': ['maintenance', 'operator'], 
     'experience_years': 5, 'base_efficiency': 1.08, 'preferred_shift': 0},
    {'id': 'W008', 'name': 'Henry_Maint', 'skills': ['maintenance'], 
     'experience_years': 7, 'base_efficiency': 1.12, 'preferred_shift': 1},
    {'id': 'W009', 'name': 'Ivy_QC', 'skills': ['quality_control', 'operator'], 
     'experience_years': 4, 'base_efficiency': 0.98, 'preferred_shift': 0},
    {'id': 'W010', 'name': 'Jack_QC', 'skills': ['quality_control'], 
     'experience_years': 3, 'base_efficiency': 0.96, 'preferred_shift': 1},
    {'id': 'W011', 'name': 'Kate_Cross', 'skills': ['operator', 'specialist_cutting'], 
     'experience_years': 5, 'base_efficiency': 1.06, 'preferred_shift': 0},
    {'id': 'W012', 'name': 'Liam_Cross', 'skills': ['operator', 'specialist_painting'], 
     'experience_years': 4, 'base_efficiency': 1.03, 'preferred_shift': 1},
    {'id': 'W013', 'name': 'Mia_Junior', 'skills': ['operator'], 
     'experience_years': 0.5, 'base_efficiency': 0.85, 'preferred_shift': 0},
    {'id': 'W014', 'name': 'Noah_Junior', 'skills': ['operator'], 
     'experience_years': 0.5, 'base_efficiency': 0.82, 'preferred_shift': 1},
    {'id': 'W015', 'name': 'Olivia_Junior', 'skills': ['operator'], 
     'experience_years': 1, 'base_efficiency': 0.88, 'preferred_shift': 0},
]
WORKER_DEFINITIONS.append({
    "id": "W016", "name": "P1",
    "skills": ["operator", "specialist_painting", "quality_control"],
    "experience_years": 3, "base_efficiency": 1.0, "preferred_shift": 0
})
WORKER_DEFINITIONS.append({
    "id": "W017", "name": "P2",
    "skills": ["operator", "specialist_painting", "quality_control"],
    "experience_years": 3, "base_efficiency": 1.0, "preferred_shift": 1
})
WORKER_POOL_CONFIG["total_workers"] = max(WORKER_POOL_CONFIG.get("total_workers", 15), 17)
MACHINE_WORKER_REQUIREMENTS = {
    'Cutting': {
        'min_workers': 1,
        'preferred_skills': ['operator', 'specialist_cutting'],
        'can_operate_alone': True,
        'quality_critical': False
    },
    'Routing': {
        'min_workers': 1, 
        'preferred_skills': ['operator'],
        'can_operate_alone': True,
        'quality_critical': False
    },
    'Painting': {
        'min_workers': 1,
        'preferred_skills': ['operator', 'specialist_painting', 'quality_control'],
        'can_operate_alone': True,
        'quality_critical': True
    },
    'Assembling': {
        'min_workers': 1,
        'preferred_skills': ['operator', 'quality_control'],
        'can_operate_alone': True,
        'quality_critical': True
    }
}
MACHINE_DEFINITIONS["TransportDock"] = {
    "capacity": 2,
    "mtbf": 1e9,
    "mttr": 0,
    "maintenance_interval": 1e9,
    "operating_cost_per_hour": 5.0,
    "maintenance_cost": 0.0,
    "breakdown_repair_multiplier": 1.0,
    "color": "gray",
    "num_machines": 1,
    "failure_model": "exponential"
}
MACHINE_WORKER_REQUIREMENTS["TransportDock"] = {
    "min_workers": 1,
    "preferred_skills": ["operator"],   
    "can_operate_alone": True,
    "quality_critical": False
}
TRANSPORTCONFIG = {
"modes": {
"forklift": {"fleet_size": 2, "speed_mps": 1.5, "capacity_units": 1, "load_time_h": 0.0056, "unload_time_h": 0.0056, "operating_cost_per_hour": 5.0, "energy_cost_per_km": 0.5},
"agv": {"fleet_size": 3, "speed_mps": 1.2, "capacity_units": 1, "load_time_h": 0.0042, "unload_time_h": 0.0042, "operating_cost_per_hour": 5.0, "energy_cost_per_km": 0.4}
},
"warehouse_node": "WH",
"floor_nodes": ["WH", "Cutting_in", "Routing_in", "Painting_in", "Assembling_in"],
"floor_edges": [
{"u": "WH", "v": "Cutting_in", "length_m": 60, "allowed_modes": ["forklift", "agv"]},
{"u": "WH", "v": "Routing_in", "length_m": 80, "allowed_modes": ["forklift", "agv"]},
{"u": "WH", "v": "Assembling_in", "length_m": 75, "allowed_modes": ["forklift", "agv"]},
{"u": "Cutting_in", "v": "Routing_in", "length_m": 40, "allowed_modes": ["conveyor"]},
{"u": "Routing_in", "v": "Painting_in", "length_m": 50, "allowed_modes": ["conveyor"]},
{"u": "Cutting_in", "v": "Painting_in", "length_m": 70, "allowed_modes": ["forklift", "agv"]}
],
"conveyors": [
{"u": "Cutting_in", "v": "Routing_in", "length_m": 40, "speed_mps": 0.5, "max_wip_units": 10},
{"u": "Routing_in", "v": "Painting_in", "length_m": 50, "speed_mps": 0.5, "max_wip_units": 10}
],
"stage_buffers": {
"Cutting": {"inbound_capacity": 50, "node": "Cutting_in"},
"Routing": {"inbound_capacity": 50, "node": "Routing_in"},
"Painting": {"inbound_capacity": 50, "node": "Painting_in"},
"Assembling": {"inbound_capacity": 50, "node": "Assembling_in"}
}
}
QUEUING_CONFIG = {
    'default_policy': {
        'policy': 'FIFO',
        'buffer_capacity': 10 
    },
    'stage_policies': {
        'Cutting': {
            'policy': 'LIFO',
            'buffer_capacity': 15
        },
        'Painting': {
            'policy': 'SPT',
            'buffer_capacity': 8
        },
        'Assembling': {
            'policy': 'ATC',
            'params': {'k': 1.0}, 
            'buffer_capacity': 12
        }
    }
}
METRICS_CONFIG = {
    "warmup_period": 60,          
    "analysis_interval": 8,       
    "bottleneck_rules": {
        "queue_length_threshold": 5,      
        "wait_time_threshold_hours": 1.0, 
        "utilization_threshold_percent": 98.0 
    },
    "alert_rules": {
        "profit_margin_threshold": 0.10,  
        "defect_rate_threshold": 0.05,    
        "oee_threshold": 0.70,            
    }
}
PLANNING_CONFIG = {
    'DEMAND_MODEL': {
        'model': 'seasonal_noise',  
        'base_per_day': 50,         
        'horizon_days': 365 * 2,    
        'rng_seed': 42,             
        'product_mix': {            
            'ProductA': 0.6,
            'ProductB': 0.4
        }
    },
    'FORECAST_MODEL': {
        'method': 'SES',  
        'alpha': 0.2      
    },
    'ABC_ANALYSIS': {
        'run_period_days': 365, 
        'a_threshold': 0.8,     
        'b_threshold': 0.95     
    }
}
