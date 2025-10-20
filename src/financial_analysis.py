import collections
from typing import Dict, Any

class FinancialAnalysis:
    """
    An advanced financial analysis engine for the factory simulation.
    This class synthesizes data from multiple trackers to produce tiered KPIs
    and data for visualizations, with a focus on correctness and depth.
    """
    def __init__(self, financial_tracker, quality_tracker, inventory_manager, machines, worker_pool, env_tracker, config, simulation_time_hours: float, total_labor_cost: float):
        # --- Core Data Trackers ---
        self.ft = financial_tracker
        self.qt = quality_tracker
        self.im = inventory_manager
        self.et = env_tracker
        self.machines = machines
        self.wp = worker_pool
        
        # --- Simulation Parameters ---
        self.config = config
        self.sim_hours = simulation_time_hours if simulation_time_hours > 0 else 1
        
        # --- Direct Costs ---
        self.total_labor_cost = total_labor_cost
        
        # --- Configuration Shortcuts for Readability ---
        self.product_configs = self.config.PRODUCTS
        self.quality_config = self.config.QUALITYCONFIG
        self.environment_config = self.config.ENVIRONMENTCONFIG
        self.machine_definitions = self.config.MACHINE_DEFINITIONS

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculates a comprehensive set of Tier 1 (operational) and Tier 2 (strategic)
        financial metrics based on the completed simulation run.

        Returns:
            A dictionary containing the calculated Tier 1 and Tier 2 metrics.
        """
        metrics = {"tier1": {}, "tier2": {}}
        
        # --- Primary Data Aggregation ---
        summary = self.ft.get_financial_summary(self.sim_hours)
        total_units_produced = len(self.ft.revenue_events)
        
        # --- Tier 1: Core Operational Financial Metrics ---

        # 1. Manufacturing Cost per Unit
        total_manufacturing_costs = summary.get('total_costs', 0)
        metrics['tier1']['manufacturing_cost_per_unit'] = {
            "label": "Manufacturing Cost per Unit",
            "value": total_manufacturing_costs / total_units_produced if total_units_produced else 0,
            "unit": "$/unit"
        }

        # 2. Cost of Quality (CoQ) - A comprehensive breakdown
        coq_breakdown = self._calculate_cost_of_quality(summary)
        metrics['tier1']['cost_of_quality'] = {
            "label": "Cost of Quality (CoQ)",
            "value": coq_breakdown['total_coq'],
            "unit": "$",
            "breakdown": coq_breakdown  # Provide detailed breakdown
        }

        # 3. Unit Contribution Margin
        total_revenue = summary.get('total_revenue', 0)
        # Variable costs are costs that scale with production (materials, machine runtime, etc.)
        # Excludes fixed costs like downtime or labor overhead.
        total_variable_costs = summary.get('total_procurement_cost', 0) + summary.get('total_machine_operating_cost', 0)
        avg_margin = (total_revenue - total_variable_costs) / total_units_produced if total_units_produced else 0
        metrics['tier1']['unit_contribution_margin'] = {
            "label": "Avg. Unit Contribution Margin",
            "value": avg_margin,
            "unit": "$/unit"
        }

        # 4. Maintenance Cost per Unit
        maint_costs = sum(e.get('total_cost', 0) for e in self.ft.downtime_losses if e.get('event_type') == 'maintenance')
        metrics['tier1']['maintenance_cost_per_unit'] = {
            "label": "Maintenance Cost per Unit",
            "value": maint_costs / total_units_produced if total_units_produced else 0,
            "unit": "$/unit"
        }

        # Placeholders for future implementation
        metrics['tier1']['overtime_rate'] = {"label": "Overtime Rate", "value": 0, "unit": "%"}
        metrics['tier1']['overtime_cost'] = {"label": "Total Overtime Cost", "value": 0, "unit": "$"}

        # --- Tier 2: Strategic Financial Metrics ---

        # 1. Inventory Turnover Rate
        cogs = total_variable_costs  # Cost of Goods Sold is approximated by total variable costs
        avg_inv_value = self._calculate_time_weighted_avg_inventory_value()
        metrics['tier2']['inventory_turnover_rate'] = {
            "label": "Inventory Turnover Rate",
            "value": cogs / avg_inv_value if avg_inv_value > 0 else 0,
            "unit": ""
        }

        # 2. Return on Assets (ROA)
        net_income = summary.get('net_profit', 0)
        total_assets = sum(
    m_def.get('purchase_price', 0) * m_def.get('nummachines', 1) 
    for m_def in self.config.MACHINE_DEFINITIONS.values()
)
        metrics['tier2']['return_on_assets'] = {
            "label": "Return on Assets (ROA)",
            "value": (net_income / total_assets) * 100 if total_assets > 0 else 0,
            "unit": "%"
        }

        # 3. Total Cost of Ownership (TCO) for production assets
        initial_purchase_price = total_assets
        total_downtime_cost = summary.get('total_downtime_cost', 0)
        grid_price = self.environment_config.get('grid_price_per_kwh', 0.12)
        total_energy_cost = self.et.total_energy_kwh * grid_price
        tco = initial_purchase_price + maint_costs + total_energy_cost + total_downtime_cost
        metrics['tier2']['total_cost_of_ownership'] = {
            "label": "Total Cost of Ownership (TCO)",
            "value": tco,
            "unit": "$"
        }

        # 4. Unused Capacity Cost (Cost of Idle Labor)
        all_machine_stats = [m.timeline.totals() for ml in self.machines.values() for m in (ml if isinstance(ml, list) else [ml])]
        total_possible_machine_hours = self.sim_hours * len(all_machine_stats)
        total_running_machine_hours = sum(s.get('running', 0) for s in all_machine_stats)
        avg_utilization = total_running_machine_hours / total_possible_machine_hours if total_possible_machine_hours else 0
        unused_capacity_cost = (1 - avg_utilization) * self.total_labor_cost
        metrics['tier2']['unused_capacity_cost'] = {
            "label": "Unused Capacity Cost",
            "value": unused_capacity_cost,
            "unit": "$"
        }

        return metrics

    def _calculate_cost_of_quality(self, financial_summary: Dict) -> Dict[str, float]:
        """Calculates the four components of Cost of Quality."""
        # 1. Appraisal Costs: Cost of inspections.
        appraisal_cost = len(self.qt.quality_events) * self.quality_config.get('inspection_cost', 0)

        # 2. Internal Failure Costs: Costs from defects found before shipping.
        scrap_costs_config = self.quality_config.get('scrap_cost_per_unit', {})
        rework_cost_config = self.quality_config.get('rework', {})
        
        internal_failure_cost = 0
        for event in self.qt.quality_events:
            if event.get('scrap'):
                prod_id = event.get('product_id')
                cost = scrap_costs_config.get(prod_id, scrap_costs_config.get('default', 0))
                internal_failure_cost += cost
            if event.get('reworked'):
                internal_failure_cost += rework_cost_config.get('rework_cost', 0)

        # 3. External Failure Costs: Costs from defects found by the customer (refunds).
        returns_config = self.quality_config.get('returns', {})
        external_failure_cost = 0
        if returns_config.get('enabled', False):
            refund_fraction = returns_config.get('refund_fraction', 1.0)
            for event in self.qt.return_events:
                prod_id = event.get('product_id')
                unit_price = self.product_configs.get(prod_id, {}).get('unit_price', 0)
                external_failure_cost += (unit_price * refund_fraction)
        
        # 4. Prevention Costs (Placeholder for future development)
        prevention_cost = 0 # Example: Could include worker training costs for quality procedures

        total_coq = appraisal_cost + internal_failure_cost + external_failure_cost + prevention_cost

        return {
            "total_coq": total_coq,
            "appraisal_costs": appraisal_cost,
            "internal_failure_costs": internal_failure_cost,
            "external_failure_costs": external_failure_cost,
            "prevention_costs": prevention_cost
        }

    def _calculate_time_weighted_avg_inventory_value(self) -> float:
        """Calculates the time-weighted average value of all inventory."""
        log = self.im.inventory_value_log
        if not log or self.sim_hours <= 0:
            return 0
        
        total_weighted_value = 0
        last_time = 0
        # Initialize with the value at the start of the simulation
        last_value = log[0][1] if log else 0

        for time, value in log:
            duration = time - last_time
            if duration > 0:
                total_weighted_value += last_value * duration
            last_time = time
            last_value = value
        
        # Account for the final period until the end of the simulation
        final_duration = self.sim_hours - last_time
        if final_duration > 0:
            total_weighted_value += last_value * final_duration
            
        return total_weighted_value / self.sim_hours

    def generate_visualization_data(self) -> Dict[str, Any]:
        """
        Generates structured data suitable for creating charts and visualizations
        in a dashboarding tool like Streamlit or Plotly.
        """
        viz_data = {}
        summary = self.ft.get_financial_summary(self.sim_hours)

        # 1. Cost Composition Breakdown
        # This provides a full picture of where money is being spent.
        coq_breakdown = self._calculate_cost_of_quality(summary)
        viz_data['cost_composition_snapshot'] = [
            {"cost_component": "Labor", "value": self.total_labor_cost},
            {"cost_component": "Raw Materials", "value": summary.get('total_procurement_cost', 0)},
            {"cost_component": "Machine Operation", "value": summary.get('total_machine_operating_cost', 0)},
            {"cost_component": "Downtime", "value": summary.get('total_downtime_cost', 0)},
            {"cost_component": "Stockouts", "value": summary.get('total_stockout_cost', 0)},
            {"cost_component": "External Failures (Refunds)", "value": coq_breakdown['external_failure_costs']},
        ]

        # 2. Pareto Analysis of Downtime Costs
        downtime_by_machine = collections.defaultdict(float)
        for event in self.ft.downtime_losses:
            if 'machine' in event and 'Worker' not in event['machine']:
                downtime_by_machine[event['machine']] += event.get('total_cost', 0)
        pareto_data = [{"machine": m, "cost": c} for m, c in downtime_by_machine.items()]
        pareto_data.sort(key=lambda x: x['cost'], reverse=True)
        viz_data['downtime_pareto'] = pareto_data

        # 3. Contribution Margin by Product
        prod_perf = summary.get('product_performance', {})
        margin_data = []
        for prod_id, data in prod_perf.items():
            margin_data.append({
                "product": prod_id,
                # Using the profit_per_unit from config as a proxy for contribution margin
                "unit_contribution_margin": self.product_configs.get(prod_id, {}).get('profit_per_unit', 0)
            })
        viz_data['contribution_margin_by_product'] = margin_data

        # 4. Maintenance vs. Failure Costs (Preventive vs. Corrective)
        maint_costs = sum(e['total_cost'] for e in self.ft.downtime_losses if e.get('event_type') == 'maintenance')
        breakdown_repair_costs = sum(e['total_cost'] for e in self.ft.downtime_losses if e.get('event_type') == 'breakdown')
        total_failure_cost = breakdown_repair_costs + coq_breakdown['internal_failure_costs'] + coq_breakdown['external_failure_costs']
        viz_data['maintenance_vs_failure_costs'] = {
            "total_preventive_maintenance_cost": maint_costs,
            "total_corrective_failure_cost": total_failure_cost
        }
        
        return viz_data

