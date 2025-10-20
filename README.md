# Advanced Factory Simulation System

This project is a high-fidelity, discrete-event simulation of a complex manufacturing facility, developed in Python using the SimPy library. It serves as a powerful digital twin for industrial engineering, operations research, and production management analysis. The system models the entire production lifecycle, from order arrival and material procurement to final product shipment, while tracking detailed operational, financial, and environmental metrics.

This simulation was developed as a research testbed to explore advanced scheduling, inventory, and quality control strategies, making it ideal for academic research and for demonstrating advanced capabilities in systems modeling and optimization.

## Key Features

The simulation incorporates a wide array of features that model real-world factory dynamics:

### Core Simulation & Process Modeling
*   **Discrete-Event Engine:** Built on **SimPy**, providing a robust foundation for modeling asynchronous processes.
*   **Multi-Product & Multi-Stage Routing:** Defines complex production paths for various products, each with unique processing times and machine requirements.
*   **Machine & Worker Dynamics:**
    *   Models machine breakdowns, scheduled maintenance, and setup times based on configurable distributions (Exponential, Weibull, Lognormal).
    *   Implements a flexible worker pool with skill-based assignments, shift schedules, and utilization tracking.
*   **Calendar & Shift Management:** A sophisticated calendar system manages 24/7 operations, including working days, multiple daily shifts, breaks, and holidays.

### Operations Research & Optimization
*   **Queuing Theory Integration:**
    *   Features analytical models for **M/M/1, M/G/1, and M/M/c queues** to compare theoretical performance against simulation results.
    *   Includes support for analyzing **Jackson Networks**.
*   **Dispatching Policies:** Implements and tests various job dispatching rules at each workstation, including:
    *   First-In, First-Out (FIFO)
    *   Shortest Processing Time (SPT)
    *   Earliest Due Date (EDD)
    *   Weighted Shortest Processing Time (WSPT)
    *   Apparent Tardiness Cost (ATC)
*   **Inventory Management:**
    *   Supports multiple inventory policies, including **(s, S) reordering** and **Economic Order Quantity (EOQ)** calculations.
    *   Tracks raw material consumption, work-in-progress (WIP), and finished goods inventory.
*   **Demand Forecasting:** Includes models to generate and compare demand forecasts against actual demand, using methods like Simple Exponential Smoothing and simulating patterns such as seasonality, trends, and random noise.

### Analytics, Dashboard & Reporting
*   **Comprehensive KPI Tracking:** Calculates over 50 key metrics, including:
    *   **Overall Equipment Effectiveness (OEE)** with detailed breakdowns of availability, performance, and quality.
    *   **Financial Analytics:** Revenue, costs, net profit, profit margin, downtime costs, and contribution margins.
    *   **Environmental Tracker:** Tracks energy consumption, emissions (grid, diesel), water usage, and waste generation based on detailed machine and material profiles.​
    *    **Quality Control Tracking:** Monitors per-stage defect rates, detection probabilities, and the impact of rework loops and scrap on overall performance.​
    *   **Throughput, cycle time, and lead time analysis.**
*   **Interactive Dashboard:** A real-time dashboard built with **Streamlit** and **FastAPI** visualizes simulation progress, key metrics, Gantt charts, and system state.
*   **Experimentation Framework:** The `experiment.py` script provides a rigorous framework for running multiple replications of different scenarios, performing statistical analysis (mean, standard deviation, confidence intervals), and generating summary reports.
*   **Data Export:** A comprehensive logging and export module saves all simulation data—including job events, machine states, and financial logs—to CSV and JSON for external analysis.

## File Structure

The project is organized into modular Python files, each with a specific responsibility:

| File                  | Description                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `main.py`             | The core simulation engine that orchestrates the entire factory model.                               |
| `config.py`           | Centralized configuration for all simulation parameters, including products, machines, and costs.      |
| `dashboard.py`        | Powers the interactive Streamlit and FastAPI dashboard for visualizing results.                       |
| `scenarios.py`        | Manages loading, modifying, and running different simulation scenarios from configuration files.       |
| `experiment.py`       | Runs controlled experiments with multiple replications and performs statistical analysis.            |
| `metrics.py`          | Calculates comprehensive KPIs, including OEE and bottleneck detection.                                  |
| `queuing_system.py`   | Implements the queuing logic, including analytical models and job queues.                                 |
| `logging_export.py`   | Handles all data logging and exporting to CSV/JSON formats.                                           |
| `financial_analysis.py` | Performs advanced financial calculations, including cost of quality and inventory turnover.        |
| `inventory_utils.py`  | Contains utility functions for inventory management, such as EOQ calculation.                         |
| `dispatch_policies.py`| Defines the various dispatching rules (SPT, EDD, ATC, etc.) for job selection.                       |
| `forecasting.py`      | Includes tools for demand generation and time-series forecasting.                                    |
| `planning.py`         | Contains logic for production planning and scheduling heuristics.                                       |

## Getting Started

### Prerequisites
- Python 3.8+
- SimPy
- Streamlit
- FastAPI
- Uvicorn
- Pandas
- NumPy

Install the required packages using:
pip install simpy streamlit fastapi uvicorn pandas numpy


### Configuration
All simulation parameters can be adjusted in the `config.py` file. This includes:
*   `SIMULATION_TIME`, `RANDOM_SEED`
*   `PRODUCTS` definitions and their routings.
*   `MACHINE_DEFINITIONS` with capacities, MTBF/MTTR, and failure models.
*   `FINANCIALCONFIG` and `QUALITYCONFIG`.
*   `CALENDAR_CONFIG` for shifts and holidays.

For more advanced configurations, you can create scenario files (e.g., in YAML format) that modify the base `config.py` for specific experiments.

### Running the Simulation

#### Single Run
To perform a single simulation run for debugging or a quick test, execute `main.py` directly:

python main.py

This will run the simulation with the default configuration and save the output to a timestamped directory in the `data/processed/` folder.

#### Running Experiments
The recommended way to run the simulation is through the experimentation framework, which provides statistical rigor.
1.  Define your scenarios in `.py` or `.yml` files.
2.  Execute `experiment.py`, passing the scenario files as arguments:

python experiment.py experiments/exp1_fifo_baseline.yaml experiments/exp1_wspt_policy.yaml -n 50

*   `experiments/exp1_.....`: Path to your scenario configuration file(s).
*   `-n 50`: The number of replications to run for each scenario (default is 10).
*   `-o experiments/`: The root directory to save all experiment results.

This will run 50 replications for each scenario, aggregate the results, and print a summary table with 95% confidence intervals for the key performance indicators.

### Viewing the Dashboard
The dashboard provides a real-time view of the simulation outputs.
1.  First, run a simulation to generate the necessary output files.
2.  Then, launch the dashboard and point it to the output directory:

streamlit run dashboard.py -- --output_dir data/processed

Alternatively, the dashboard can be served as an API using FastAPI:
python dashboard.py --serve --output_dir data/processed
The API will be available at `http://127.0.0.1:8000`.
