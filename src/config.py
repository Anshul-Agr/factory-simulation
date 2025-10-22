# ======================================================================================
# MASTER CONFIGURATION FILE for Factory Simulation
# ======================================================================================
# This file centralizes all parameters for the simulation. By modifying this file,
# one can control every aspect of the simulation without altering the core source code.
#
# Each section is documented with available options based on the project's codebase.
# ======================================================================================

# IMPORTS
import math
from pathlib import Path
from dispatch_policies import DispatchPolicyBase, get_dispatch_policy

# --------------------------------------------------------------------------------------
# 1. CORE SIMULATION & SCENARIO CONTROL
# --------------------------------------------------------------------------------------
# High-level parameters that define the simulation's execution scope and operational mode.
# Used by: main.py run_simulation()
# --------------------------------------------------------------------------------------

# DATA SOURCE

USE_CSV_ORDERS = True # FALSE only for Hard-Coded demand
BASE_DIR = Path(__file__).resolve().parents[1] # TO ENTER PARENT DIRECTORY
CSV_ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv" # Can be changed to any desired path, I chose data/raw/orders.csv for demonstration

# SIMULATION BASICS

SIMULATION_TIME = 500 # IN HOURS
RANDOM_SEED = 42       
WARMUP_PERIOD = 0 # Initial period (in hours) to run the simulation before collecting metrics which allows system to get into a steady state

# = SCHEDULING
# Core scheduling logic for the entire factory.
    # Options:
    # - 'EVENT_BASED': (Default) Jobs are processed as they arrive based on resource availability.
    # - 'STATIC_PLAN': Jobs are scheduled in advance using a specified heuristic.

# Heuristic used to sequence jobs when SCHEDULING_MODE is 'STATIC_PLAN'.
    # Options : 'cds', 'palmer'.

SCHEDULING_MODE = 'EVENT_BASED' 
PLANNING_HEURISTIC = 'cds' 

# --- MODE 1: Manual CSV File (Your Default) ---
# Use this for hand-crafted, specific scenarios.
# The simulation will load orders directly from the file path you provide.
## Make sure USE_CSV_ORDERS is also True.


# --- MODE 2: Generated Demand to a NEW CSV File (Recommended for Experiments) ---
# Use this to create complex, repeatable scenarios for testing.
# The simulation will first CREATE a new CSV file based on the demand model
# and then load it using your existing CSV logic. You can review/edit the
# generated file at `DEMAND_CONFIG['csv_output_path']` before running.
#eg. 
# ORDER_SOURCE = "DEMAND_TO_CSV"
# DEMAND_CONFIG = {
#     "horizon_days": 90,
#     "base_per_day": 10,           # Total units demanded per day, across all products.
#     "model": "seasonal_noise",
#     "rng_seed": 42,
#     "csv_output_path": "data/raw/orders/generated_demand_orders.csv",
    
#     # --- NEW: Define the mix of products to generate ---
#     "product_mix": {
#         "ProductA": 0.4,                # 40% of orders will be for ProductA
#         "ProductB": 0.3,                # 30% of orders will be for ProductB
#         "ProductD_with_Assembly": 0.2,  # 20% of orders will be for ProductD
#         "Oak_Dining_Table": 0.1         # 10% of orders will be for Oak_Dining_Table
#     },
    
#     # These are now defaults for any product not specified in the mix
#     "due_lag_days": 7,
#     "per_unit_orders": True
# }


# --- MODE 3: Legacy Hard-Coded Dictionary ---
# Use this to run the hard-coded DAILY_ORDERS dictionary.
#
# ORDER_SOURCE = "DAILY"
# Make sure USE_CSV_ORDERS is set to False.


# =============================================================================
ORDER_SOURCE = "DYNAMIC" 
DEMAND_CONFIG = {
    "horizon_days": 90,
    "base_per_day": 10,         # Total units demanded per day, across all products.  
    "model": "seasonal_noise",  # random_walk, seasonal_spikes or forecast_bias ALSO AVAILABLE
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

# CHANGE FINANCIAL PARAMETERS BASED PER HOUR/UNIT

FINANCIAL_CONFIG = {
    'labor_cost_per_hour': 20,      
    'downtime_cost_per_hour': 40,
    'stockout_cost_per_unit': 15.0,
}

# CALENDAR CONFIGURATION TO DEFINE WORKING DAYS AND HOURS

CALENDAR_CONFIG = {
    # General factory schedule
    'week_length': 7 * 24,
    'working_days': [0, 1, 2, 3, 4, 5],  # 0=Monday to 5=Saturday
    'daily_shifts': [
        {'start': 6,  'end': 14, 'breaks': [(10, 10.5)]},  # Morning shift
        {'start': 14, 'end': 22, 'breaks': [(18, 18.5)]},  # Afternoon shift
    ],
    'holidays': [],  # List of holiday dates as strings, e.g., ["2025-12-25"]

    # --- Machine-Specific Calendar Overrides ---
    # Defines exceptions to the general factory schedule for specific machine STAGES.
    'machine_overrides': {
        'Painting': {
            "daily_shifts": [
                {'start': 6,  'end': 14, 'breaks': [(10, 10.5)]},
                {'start': 15, 'end': 23, 'breaks': [(18, 18.5)]},
            ],
            "working_days": [0, 1, 2, 3, 4, 5]
        },
        # EXAMPLE of a true override for a stage that runs 24/7:
        # 'CNC_Machining': {
        #     'working_days': [0, 1, 2, 3, 4, 5, 6],
        #     'daily_shifts': [{'start': 0, 'end': 24, 'breaks': []}]
        # }
    },

    # --- Scheduled Preventive Maintenance ---
    # Defines recurring maintenance windows for specific, individual MACHINES.
    'maintenance_windows': {
        "Painting_Machine1": {"every_hours": 500, "offset": 0},
        "Painting_Machine2": {"every_hours": 500, "offset": 125},
        "Painting_Machine3": {"every_hours": 500, "offset": 250}
    }
}

# USE THIS TO CHANGE THE SHIPPING TIME, EASY TO DISABLE/ENABLE SHIPPING PARTIAL ORDERS
SHIPPING_CONFIG = {
    'daily_shipping_hour': 19.0,   
    'ship_partial_orders': True    
}

# Comprehensive ENVIRONMENTAL Description and Configuration

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

# ADVANCED QUALITY CONFIGURATION 

QUALITYCONFIG = {
    "enabled": True,
    "defect_p": {
        # Default baseline per stage if product-specific not provided
        "default": {"Cutting": 0.01, "Routing": 0.015, "Painting": 0.03, "Assembling": 0.025},
        # Product overrides (only list stages that differ from default)
        "ProductA": {"Painting": 0.025},
        "ProductB": {},
        "ProductC": {},
        "ProductD_with_Assembly": {"Assembling": 0.03}
    },
    # Detection probability depending on QC presence at stage
    "detect_p": {"with_qc": 0.9, "no_qc": 0.6},
    # Rework controls (we will enable after verification; initially enabled True but success prob set safely)
    "rework": {"enabled": True, "max_loops": 1, "time_factor": 0.5, "success_p": 0.7},
    # Cost if scrapped (used by FinancialTracker adjustments)
    "scrap_cost_per_unit": {"default": 10.0, "ProductA": 12.0, "ProductD_with_Assembly": 18.0},
    # Customer returns for escaped defects
    "returns": {"enabled": True, "escape_to_return_p": 0.5, "delay_h": 24.0, "refund_fraction": 1.0}
}

# --- FLOW CONTROL (CONWIP) CONFIGURATION ---

FLOW_CONTROL_CONFIG = {
    'enabled': True,  # Set to True to activate the WIP cap, False to run as a push system.
    
    # This is the CONWIP level: the maximum number of jobs allowed in the
    # entire system (from first process to finished goods) at any one time.
    'wip_limit': 20
}

# ==============================================================================
# --- SEQUENCE-DEPENDENT SETUP TIMES ---
# ==============================================================================
# Defines the setup time (in hours) when changing from one product to another.
# The format is {from_product: {to_product: time}}.
# If a specific from-to pair is not defined, the simulation will use the
# default setup time defined for the product-stage combination in PRODUCTS.


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

# PRODUCT DEFINITION AND CONFIGURATION

PRODUCTS = {
    'ProductA': {
        'routing': ['Cutting', 'Routing', 'Painting'],
        'machine_times': {'Cutting': (3, 4), 'Routing': (4, 5), 'Painting': (5, 6)},
        'load_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'unload_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'setup_times': {'Cutting': 1, 'Routing': 2, 'Painting': 1},
        'color': 'skyblue',
        'unit_price': 900

        # Optional ideal times for performance calc:
        # 'ideal_cycle_times': {'Cutting': 3.5, 'Routing': 4.5, 'Painting': 5.5}
    },
    'ProductB': {
        'routing': ['Routing', 'Cutting', 'Painting'],
        'machine_times': {'Cutting': (2, 3), 'Routing': (3, 6), 'Painting': (4, 5)},
        'load_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'unload_times': {'Cutting': (1, 2), 'Routing': (1, 2), 'Painting': (1, 2)},
        'setup_times': {'Cutting': 2, 'Routing': 1, 'Painting': 2},
        'color': 'salmon',
        'unit_price': 1000
    },
    'ProductC': {
        'routing': ['Cutting', 'Painting'],
        'machine_times': {'Cutting': (4, 5), 'Painting': (6, 7)},
        'load_times': {'Cutting': (1, 1), 'Painting': (2, 2)},
        'unload_times': {'Cutting': (1, 1), 'Painting': (2, 2)},
        'setup_times': {'Cutting': 1.5, 'Painting': 1.5},
        'color': 'lightgreen',
        'unit_price': 1200
    },
    'ProductD_with_Assembly': {
        'routing': ['Cutting', 'Assembling', 'Painting'],
        'machine_times': {'Cutting': (2, 3), 'Assembling': (5, 8), 'Painting': (4, 5)},
        'load_times': {'Cutting': (1, 2), 'Assembling': (1, 1), 'Painting': (1, 2)},
        'unload_times': {'Cutting': (1, 2), 'Assembling': (1, 1), 'Painting': (1, 2)},
        'setup_times': {'Cutting': 1, 'Assembling': 2, 'Painting': 1.5},
        'color': 'gold',
        'unit_price': 1500
    }
}

# HIGHLY CONFIGURABLE MACHINE DEFINITIONS AND PARAMETERS
MACHINE_DEFINITIONS = {
'Cutting': {
    'capacity': 1,                      # Number of jobs one machine can process at the same time.
    'mtbf': 300,                        # Mean Time Between Failures (in hours).
    'mttr': 4,                          # Mean Time To Repair (in hours) after a failure.
    'maintenance_interval': 600,        # Hours between scheduled preventive maintenance events.
    'purchase_price': 50000,            # PURCHASE price for ROA calculation
    'operating_cost_per_hour': 5,       # Cost incurred for every hour the machine is running.
    'maintenance_cost': 120,            # Fixed cost for one preventive maintenance cycle.
    'breakdown_repair_multiplier': 2.0, # Multiplier for maintenance_cost to calculate breakdown repair cost.
    'color': 'red',                     # Color used for Gantt chart visualization.
    'num_machines': 4,                  # Number of identical parallel machines at this station.
    'failure_model': 'exponential',     # Statistical model for failures ('exponential', 'weibull', 'lognormal').
    "analytical_model": {
            "type": "M/M/1",            # Kendall's notation for the theoretical queuing model.
            "arrival_rate_lambda": 0.2, # Avg. job arrivals per hour (for manual calculation).
            "service_rate_mu": 0.25,    # Avg. jobs completed per hour by one machine (for manual calculation).
            "arrival_scv": 1.0,         # Squared Coefficient of Variation for arrivals (not used in M/M/1).
            "service_scv": 1.0          # Squared Coefficient of Variation for service times (not used in M/M/1).
    }},
'Routing': {
    'capacity': 1,
    'mtbf': 400,
    'mttr': 6,
    'maintenance_interval': 700,
    'purchase_price': 70000,
    'operating_cost_per_hour': 8,
    'maintenance_cost': 150,
    'breakdown_repair_multiplier': 1.8,
    'color': 'blue',
    'num_machines': 4,
    'failure_model': 'weibull',         # Weibull model can represent increasing failure rate over time (wear-out).
    'weibull_k': 2.0,                   # Weibull shape parameter (k > 1 implies wear-out).
    'weibull_lambda': 450,              # Weibull scale parameter (characteristic life).
    "analytical_model": {
            "type": "M/G/1",            # M/G/1: Model with General (non-exponential) service times.
            "arrival_rate_lambda": 0.18,
            "service_rate_mu": 0.22,
            "arrival_scv": 1.0,
            "service_scv": 1.0          # SCV for service is used in M/G/1 calculations.
    }},
'Painting': {
    'capacity': 1,
    'mtbf': 320,
    'mttr': 5,
    'maintenance_interval': 500,
    'purchase_price': 30000,
    'operating_cost_per_hour': 2,
    'maintenance_cost': 140,
    'breakdown_repair_multiplier': 2.2,
    'color': 'green',
    'num_machines': 4,
    'failure_model': 'lognormal',       # Lognormal model is good for failures caused by fatigue/degradation.
    'lognorm_mu': 5.5,                  # Mean of the log of the failure times.
    'lognorm_sigma': 0.25,              # Standard deviation of the log of the failure times.
    "analytical_model": {
            "type": "M/M/c",            # M/M/c: Model with 'c' parallel servers (num_machines).
            "arrival_rate_lambda": 0.5, # Total arrival rate to the entire station.
            "service_rate_mu": 0.2,     # Service rate for each individual machine.
            "arrival_scv": 1.0,
            "service_scv": 1.0
    }},
'Assembling': {
    'capacity': 1,
    'mtbf': 150,
    'mttr': 8,
    'maintenance_interval': 400,
    'purchase_price': 80000,
    'operating_cost_per_hour': 5,
    'maintenance_cost': 200,
    'breakdown_repair_multiplier': 1.5,
    'color': 'purple',
    'num_machines': 4,
    'failure_model': 'weibull',
    'weibull_k': 1.5,
    'weibull_lambda': 180,
    "analytical_model": {
            "type": "M/M/c",            # The queuing model to use for this station.
            "arrival_rate_lambda": 0.3,
            "service_rate_mu": 0.15,
            "arrival_scv": 1.0,
            "service_scv": 1.0
}
}}

# ======================================================================================
# DEFAULT MACHINE CONFIGURATION
# ======================================================================================
# Fallback values for any stage/machine defined in a product's routing
# that does not have an explicit entry in MACHINE_DEFINITIONS above.
# ======================================================================================

DEFAULT_MACHINE_CONFIG = {
    'capacity': 1,
    'mtbf': 300,
    'mttr': 6,
    'maintenance_interval': 600,
    'purchase_price': 50000,
    'operating_cost_per_hour': 5,
    'maintenance_cost': 150,
    'breakdown_repair_multiplier': 1.5,
    'color': 'gray',
    'num_machines': 2,
    'failure_model': 'exponential'
}

# ================================ ENERGY AND ENVIRONMENTAL FACTORS ====== #

MACHINE_ENERGY_PROFILES = {
  "Cutting": {
    "running_kW": 15.0,             # Power draw (kW) when the machine is actively processing a job.
    "setup_kW": 10.0,               # Power draw during a changeover or setup procedure.
    "idle_kW": 4.0,                 # Power draw when the machine is on but not processing (awaiting a job).
    "waitingworker_kW": 3.0,        # Power draw when the machine has a job but is waiting for an operator.
    "waitingmaterial_kW": 3.0,      # Power draw when the machine is waiting for raw materials.
    "maintenance_kW": 2.5,          # Power draw during a scheduled preventive maintenance event.
    "breakdown_kW": 2.0,            # Power draw when the machine is non-operational due to a failure.
    "calendaroff_kW": 0.05          # Standby or parasitic power draw when the machine is off due to schedule.
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
    "running_kW": 20.0,             # Higher power consumption typical of painting/curing processes.
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
    "running_kW": 4.0,              # Energy profile for equipment at the shipping/receiving dock.
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
    "energy_mode": "diesel",        # Specifies the type of energy source used.
    "l_per_km": 0.08,               # Fuel consumption in Liters per kilometer for diesel vehicles.
    "kwh_per_km": 0.0               # Electrical consumption in kWh per kilometer (zero for diesel).
  },
  "agv": {
    "energy_mode": "electric",      # Automated Guided Vehicle running on electricity.
    "l_per_km": 0.0,                # Fuel consumption (zero for electric).
    "kwh_per_km": 0.5               # Electrical energy consumption per kilometer.
  },
  "conveyor": {
    "energy_mode": "electric",      # Conveyor belt system running on electricity.
    "kwh_per_km": 1.0,              # Electrical consumption, often rated per unit of distance moved.
    "l_per_km": 0.0
  }
}

PRODUCT_ENV = {
  "ProductA": {"waste_kg": 0.2, "recycle_rate": 0.5, "water_m3": 0.05},      # Environmental impact metrics specific to each finished product.
  "ProductB": {"waste_kg": 0.25, "recycle_rate": 0.45, "water_m3": 0.06},     # waste_kg: Scrap generated per unit. recycle_rate: Fraction of scrap recycled.
  "ProductC": {"waste_kg": 0.3, "recycle_rate": 0.4, "water_m3": 0.04},       # water_m3: Water consumed per unit produced (in cubic meters).
  "ProductD_with_Assembly": {"waste_kg": 0.35, "recycle_rate": 0.5, "water_m3": 0.08}
}

MATERIAL_ENV = {
  "Steel_Sheet": {"waste_factor": 0.05, "water_m3_per_unit": 0.001},       # Environmental impact associated with the consumption of raw materials.
  "Paint": {"waste_factor": 0.03, "water_m3_per_unit": 0.0},                 # waste_factor: The fraction of the material that becomes scrap during use.
  "Routing_Bits": {"waste_factor": 0.02, "water_m3_per_unit": 0.0},        # water_m3_per_unit: Water consumed per unit of the raw material itself.
  "Assembly_Parts": {"waste_factor": 0.01, "water_m3_per_unit": 0.0005}
}

#===========================================================#

# ADVANCED QUEUE ANALYTICS

ANALYTICAL_QUEUE_CONFIG = {
    "enabled": True,
    "jackson_network_enabled": True,
    "parameter_source": "auto" 
}

# HARD-CODED DAILY ORDERS, Work with the daily setting in ORDER SOURCE


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


RAW_MATERIALS = {
    'Steel_Sheet': {
        'initial_stock': 150,                       # The starting quantity of this material at the beginning of the simulation.
        'cost_per_unit': 5.0,                       # The base cost to the factory for one unit of this material.
        'used_by_stages': ['Cutting']               # A list of machine stages that consume this material (for reference/validation).
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
    'storage_capacity': 1000,                       # Total capacity of the raw material warehouse.
    'Steel_Sheet': {
        'policy': 's_S',                            # Replenishment policy: when stock <= s, order up to S.
        's': 75,                                    # The reorder point (lower bound).
        'S': 200,                                   # The order-up-to level (upper bound).
        'available_suppliers': ['Supplier_NA_Reliable', 'Supplier_EU_Standard'] # List of potential suppliers for this material.
    },
    'Paint': {
        'policy': 'BASE_STOCK',                     # Replenishment policy: At fixed intervals, order up to a target level.
        'review_period_h': 24,                      # How often (in hours) to check the stock level.
        'target_level': 100,                        # The stock level to order up to during a review.
        'available_suppliers': ['Supplier_EU_Standard', 'Supplier_Asia_Unstable']
    },
    'Routing_Bits': {
        'policy': 'REORDER_POINT',                  # Replenishment policy: When stock hits a reorder point, order a fixed quantity.
        'reorder_level': 40,                        # The stock level that triggers a new order.
        'order_qty': 100,                           # The fixed quantity to order each time.
        'available_suppliers': ['Supplier_NA_Reliable']
    },
    'Assembly_Parts': {
        'policy': 'EOQ',                            # Economic Order Quantity: Dynamically calculates the optimal order size.
        'params_eoq': { 'safety_stock_days': 5 },   # Parameters specific to the EOQ calculation, like safety stock coverage.
        'available_suppliers': ['Supplier_NA_Reliable']
    }
}
SUPPLY_CHAIN_CONFIG = {
    'enabled': True,                                # Global switch to turn the supply chain simulation on or off.
    'risk_model': 'dynamic_events',                 # Method for simulating risks: 'dynamic_events' uses disruption events while 'simple_stochastic' has no interference
    'procurement_policy': 'best_reliability',       # Logic for supplier selection: 'best_reliability' or 'lowest_cost'.
    'suppliers': {
        'Supplier_NA_Reliable': {
            'region': 'North America',              # Geographical region for targeting disruptions.
            'reliability': 0.995,                   # Base probability (0 to 1) that a delivery will arrive successfully.
            'lead_time_dist': {'type': 'normal', 'mean': 24, 'stddev': 4}, # Statistical distribution of delivery lead time in hours.
            'cost_multiplier': 1.2,                 # A factor applied to the material's base cost (e.g., 1.2 = 20% premium).
            'min_order_qty': 50                     # Minimum Order Quantity that this supplier will accept.
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
    'disruption_events': [                           # A list of discrete events that temporarily affect suppliers in a region.
        {
            'name': 'Major Port Strike',            # Name of the event for logging.
            'target_region': 'Asia',                # Which supplier region this event impacts.
            'start_time': 500,                      # When the event starts (in simulation hours).
            'duration': 20,                         # How long the event lasts (in simulation hours).
            'impact': { 'reliability_multiplier': 0.01, 'lead_time_multiplier': 3.0 } # Factors applied to supplier stats during the event.
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
        'Cutting': {'Steel_Sheet': 2},              # To make ProductA, the 'Cutting' stage consumes 2 units of 'Steel_Sheet'.
        'Routing': {'Routing_Bits': 0.1},           # The 'Routing' stage consumes 0.1 units of 'Routing_Bits'.
        'Painting': {'Paint': 1}                    # The 'Painting' stage consumes 1 unit of 'Paint'.
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
    'total_workers': 15,                # The total number of workers available in the pool.
    'shift_overlap_minutes': 30,        # Duration (in minutes) for hand-off between outgoing and incoming shifts.
    'fatigue_enabled': True,            # If True, worker efficiency decreases over time as they work.
    'fatigue_recovery_rate': 0.2,       # Rate at which worker fatigue level decreases when they are not working.
    'max_fatigue_threshold': 0.95,      # A fatigue level (0-1) above which a worker might refuse work or be less efficient.
    'unpredictability_enabled': False,  # If True, enables random events like sick days and vacations.
}
WORKER_SKILLS = {
    'operator': {
        'description': 'Basic machine operation',   # A human-readable description of the skill.
        'compatible_machines': ['Cutting', 'Routing', 'Painting', 'Assembling'], # List of machine stages this skill can operate.
        'efficiency_multiplier': 1.0,           # A multiplier affecting how quickly a worker with this skill completes a task.
        'training_time': 40                     # Time (in hours) required to train a worker in this skill.
    },
    'maintenance': {
        'description': 'Machine maintenance and repair',
        'compatible_machines': ['Cutting', 'Routing', 'Painting', 'Assembling'],
        'efficiency_multiplier': 1.2,           # Higher efficiency for specialized tasks.
        'training_time': 80
    },
    'quality_control': {
        'description': 'Quality inspection and control',
        'compatible_machines': ['Painting', 'Assembling'],
        'efficiency_multiplier': 0.9,           # QC tasks might be slower/more deliberate, hence < 1.0.
        'training_time': 60
    },
    'specialist_cutting': {
        'description': 'Advanced cutting operations',
        'compatible_machines': ['Cutting'],
        'efficiency_multiplier': 1.3,           # Specialist skills provide a significant efficiency boost.
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
    'algorithm': 'skill_priority',              # The logic used by the WorkerPool to assign an available worker to a task.
    'priority_weights': {
        'skill_match': 0.35,                    # The weight given to how well a worker's skills match the task requirements.
        'efficiency': 0.4,                      # The weight given to the worker's overall efficiency.
        'fatigue_level': 0.15,                  # The weight given to preferring less fatigued workers.
        'experience': 0.1                       # The weight given to a worker's years of experience.
    },
    'max_consecutive_hours': 12,                # The maximum number of hours a worker can be assigned tasks before a mandatory break.
    'mandatory_break_hours': 1,                 # The duration of the mandatory break after hitting max_consecutive_hours.
}
WORKER_UNPREDICTABILITY = {
    'sick_day_probability': 0.02,               # The daily probability that a worker will call in sick.
    'vacation_probability': 0.001,              # The daily probability that a worker will start a pre-planned vacation.
    'sick_duration_dist': {'type': 'discrete', 'values': [1, 2, 3, 5], 'probs': [0.5, 0.3, 0.15, 0.05]}, # Distribution of sick leave duration in days.
    'vacation_duration_dist': {'type': 'discrete', 'values': [5, 10, 15], 'probs': [0.6, 0.3, 0.1]}, # Distribution of vacation duration in days.
    'late_arrival_probability': 0.0001,         # The probability that a worker will be late for their shift.
    'late_arrival_delay_dist': {'type': 'uniform', 'low': 0.25, 'high': 2.0} # Distribution of lateness duration in hours.
}

WORKER_DEFINITIONS = [
    {'id': 'W001', 'name': 'Alice_Senior', 'skills': ['operator', 'maintenance', 'specialist_cutting'],
     'experience_years': 8, 'base_efficiency': 1.1, 'preferred_shift': 0}, # Senior worker with multiple high-value skills.
    {'id': 'W002', 'name': 'Bob_Senior', 'skills': ['operator', 'quality_control', 'specialist_painting'],
     'experience_years': 6, 'base_efficiency': 1.05, 'preferred_shift': 0}, # Senior worker with a focus on painting and quality.
    {'id': 'W003', 'name': 'Carol_Op', 'skills': ['operator'],
     'experience_years': 3, 'base_efficiency': 1.0, 'preferred_shift': 0}, # An experienced standard operator.
    {'id': 'W004', 'name': 'David_Op', 'skills': ['operator'],
     'experience_years': 2, 'base_efficiency': 0.95, 'preferred_shift': 1}, # A less experienced operator on the second shift.
    {'id': 'W005', 'name': 'Eve_Op', 'skills': ['operator', 'quality_control'],
     'experience_years': 4, 'base_efficiency': 1.02, 'preferred_shift': 1}, # An operator cross-trained in quality control.
    {'id': 'W006', 'name': 'Frank_Op', 'skills': ['operator'],
     'experience_years': 1, 'base_efficiency': 0.9, 'preferred_shift': 1}, # A junior operator.
    {'id': 'W007', 'name': 'Grace_Maint', 'skills': ['maintenance', 'operator'],
     'experience_years': 5, 'base_efficiency': 1.08, 'preferred_shift': 0}, # A maintenance specialist who can also operate machines.
    {'id': 'W008', 'name': 'Henry_Maint', 'skills': ['maintenance'],
     'experience_years': 7, 'base_efficiency': 1.12, 'preferred_shift': 1}, # A highly experienced, dedicated maintenance technician.
    {'id': 'W009', 'name': 'Ivy_QC', 'skills': ['quality_control', 'operator'],
     'experience_years': 4, 'base_efficiency': 0.98, 'preferred_shift': 0}, # A quality control specialist who can also operate machines.
    {'id': 'W010', 'name': 'Jack_QC', 'skills': ['quality_control'],
     'experience_years': 3, 'base_efficiency': 0.96, 'preferred_shift': 1}, # A dedicated quality control specialist.
    {'id': 'W011', 'name': 'Kate_Cross', 'skills': ['operator', 'specialist_cutting'],
     'experience_years': 5, 'base_efficiency': 1.06, 'preferred_shift': 0}, # A cross-trained cutting specialist.
    {'id': 'W012', 'name': 'Liam_Cross', 'skills': ['operator', 'specialist_painting'],
     'experience_years': 4, 'base_efficiency': 1.03, 'preferred_shift': 1}, # A cross-trained painting specialist.
    {'id': 'W013', 'name': 'Mia_Junior', 'skills': ['operator'],
     'experience_years': 0.5, 'base_efficiency': 0.85, 'preferred_shift': 0}, # A new hire with low base efficiency.
    {'id': 'W014', 'name': 'Noah_Junior', 'skills': ['operator'],
     'experience_years': 0.5, 'base_efficiency': 0.82, 'preferred_shift': 1}, # A new hire on the second shift.
    {'id': 'W015', 'name': 'Olivia_Junior', 'skills': ['operator'],
     'experience_years': 1, 'base_efficiency': 0.88, 'preferred_shift': 0}, # A junior worker with one year of experience.
    {
    'id': 'W016', 'name': 'P1',
    'skills': ['operator', 'specialist_painting', 'quality_control'],
    'experience_years': 3, 'base_efficiency': 1.0, 'preferred_shift':0      # A versatile, multi-skilled worker.
}
]

# This line dynamically ensures the 'total_workers' parameter is at least as large as the number of workers defined above.
WORKER_POOL_CONFIG["total_workers"] = max(WORKER_POOL_CONFIG.get("total_workers", 15), 17)

MACHINE_WORKER_REQUIREMENTS = {
    'Cutting': {
        'min_workers': 1,                                       # Minimum number of workers required to operate the machine.
        'preferred_skills': ['operator', 'specialist_cutting'], # The worker assignment algorithm will prioritize workers with these skills.
        'can_operate_alone': True,                              # If True, a single worker can manage the machine.
        'quality_critical': False                               # If True, might influence the need for a QC-skilled worker.
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
        'quality_critical': True                                # Painting is marked as a quality-critical step.
    },
    'Assembling': {
        'min_workers': 1,
        'preferred_skills': ['operator', 'quality_control'],
        'can_operate_alone': True,
        'quality_critical': True                                # Assembly is also marked as a quality-critical step.
    }
}


MACHINE_DEFINITIONS["TransportDock"] = {
    "capacity": 2,                                  # Can handle two transport jobs (e.g., loading/unloading) at once.
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
"forklift": {"fleet_size": 2, "speed_mps": 1.5, "capacity_units": 1, "load_time_h": 0.0056, "unload_time_h": 0.0056, "operating_cost_per_hour": 5.0, "energy_cost_per_km": 0.5}, # Defines a pool of 2 forklifts
"agv": {"fleet_size": 3, "speed_mps": 1.2, "capacity_units": 1, "load_time_h": 0.0042, "unload_time_h": 0.0042, "operating_cost_per_hour": 5.0, "energy_cost_per_km": 0.4} # Defines a pool of 3 Automated Guided Vehicles
},
"warehouse_node": "WH", # The designated name for the main warehouse location in the layout graph.
"floor_nodes": ["WH", "Cutting_in", "Routing_in", "Painting_in", "Assembling_in"], # A list of all valid locations (nodes) in the factory layout.
"floor_edges": [
{"u": "WH", "v": "Cutting_in", "length_m": 60, "allowed_modes": ["forklift", "agv"]}, # Path from Warehouse to Cutting, 60m long, accessible by forklift/AGV.
{"u": "WH", "v": "Routing_in", "length_m": 80, "allowed_modes": ["forklift", "agv"]}, # Path from Warehouse to Routing.
{"u": "WH", "v": "Assembling_in", "length_m": 75, "allowed_modes": ["forklift", "agv"]}, # Path from Warehouse to Assembling.
{"u": "Cutting_in", "v": "Routing_in", "length_m": 40, "allowed_modes": ["conveyor"]}, # Path from Cutting to Routing, only accessible by a conveyor.
{"u": "Routing_in", "v": "Painting_in", "length_m": 50, "allowed_modes": ["conveyor"]}, # Path from Routing to Painting via conveyor.
{"u": "Cutting_in", "v": "Painting_in", "length_m": 70, "allowed_modes": ["forklift", "agv"]} # Alternative path from Cutting to Painting via forklift/AGV.
],
"conveyors": [
{"u": "Cutting_in", "v": "Routing_in", "length_m": 40, "speed_mps": 0.5, "max_wip_units": 10}, # Defines a specific conveyor with its own speed and capacity.
{"u": "Routing_in", "v": "Painting_in", "length_m": 50, "speed_mps": 0.5, "max_wip_units": 10}
],
"stage_buffers": {
"Cutting": {"inbound_capacity": 50, "node": "Cutting_in"}, # Associates the 'Cutting' stage with a layout node and defines its input buffer size.
"Routing": {"inbound_capacity": 50, "node": "Routing_in"},
"Painting": {"inbound_capacity": 50, "node": "Painting_in"},
"Assembling": {"inbound_capacity": 50, "node": "Assembling_in"}
}
}
QUEUING_CONFIG = {
    'default_policy': {
        'policy': 'FIFO',           # First-In, First-Out: The default rule for any station not specified below.
        'buffer_capacity': 10       # Default number of jobs that can wait in a machine's queue.
    },
    'stage_policies': {
        'Cutting': {
            'policy': 'LIFO',       # Last-In, First-Out: Prioritizes the most recent arrivals.
            'buffer_capacity': 15
        },
        'Painting': {
            'policy': 'SPT',        # Shortest Processing Time: Prioritizes jobs that take the least time to complete.
            'buffer_capacity': 8
        },
        'Assembling': {
            'policy': 'ATC',        # Apparent Tardiness Cost: A dynamic rule considering due date, processing time, and profit.
            'params': {'k': 1.0},   # The 'k' parameter tunes the look-ahead factor for the ATC policy.
            'buffer_capacity': 12
        }
    }
}
METRICS_CONFIG = {
    "warmup_period": 60,            # Duration (in hours) to run the simulation before starting metric collection.
    "analysis_interval": 8,         # How often (in hours) the bottleneck and alert analysis should run.
    "bottleneck_rules": {
        "queue_length_threshold": 5,        # A queue length exceeding this value can trigger a bottleneck flag.
        "wait_time_threshold_hours": 1.0,   # A job wait time exceeding this value can trigger a bottleneck flag.
        "utilization_threshold_percent": 98.0 # Machine utilization exceeding this can trigger a bottleneck flag.
    },
    "alert_rules": {
        "profit_margin_threshold": 0.10,    # If overall profit margin drops below 10%, an alert is triggered.
        "defect_rate_threshold": 0.05,      # If the defect rate for any product exceeds 5%, an alert is triggered.
        "oee_threshold": 0.70,              # If the factory's Overall Equipment Effectiveness (OEE) drops below 70%, an alert is triggered.
    }
}

# ADVANCED INVENTORY CLASSIFICATION AND FORECAST PLANNING


PLANNING_CONFIG = {
    'DEMAND_MODEL': {
        'model': 'seasonal_noise',    # The pattern used to generate dynamic demand ('seasonal_noise', 'random_walk', 'seasonal_spikes' or 'forecast_bias').
        'base_per_day': 50,           # The average number of units demanded per day.
        'horizon_days': 365 * 2,      # The time horizon over which to generate the demand pattern.
        'rng_seed': 42,               # A separate random seed for demand generation to ensure its reproducibility.
        'product_mix': {              # The proportion of total demand allocated to each product.
            'ProductA': 0.6,
            'ProductB': 0.4
        }
    },
    'FORECAST_MODEL': {
        'method': 'SES',            # The time-series forecasting method to use (e.g., Simple Exponential Smoothing).
        'alpha': 0.2                # The smoothing factor for SES (a value between 0 and 1).
    },
    'ABC_ANALYSIS': {
        'run_period_days': 365,     # The interval between which to conduct ABC ANALYSIS
        'a_threshold': 0.8,         # Items representing the top 80% of value are classified as 'A'.
        'b_threshold': 0.95         # Items representing the next 15% of value (up to 95%) are 'B'.
    }
}
