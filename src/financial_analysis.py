import collections
class FinancialAnalysis:
    def __init__(self, financial_tracker, quality_tracker, inventory_manager, machines, worker_pool, env_tracker, config, simulation_time_hours, total_labor_cost):
        self.ft = financial_tracker
        self.qt = quality_tracker
        self.im = inventory_manager
        self.machines = machines
        self.wp = worker_pool
        self.et = env_tracker
        self.config = config
        self.sim_hours = simulation_time_hours if simulation_time_hours > 0 else 1
        self.total_labor_cost = total_labor_cost
        self.product_configs = self.config.PRODUCTS
        self.quality_config = self.config.QUALITYCONFIG
    def _calculate_time_weighted_avg_inventory_value(self):
        log = self.im.inventory_value_log
        if not log or self.sim_hours <= 0:
            return 0
        total_weighted_value = 0
        last_time = 0
        last_value = log[0][1]
        for time, value in log:
            duration = time - last_time
            if duration > 0:
                total_weighted_value += last_value * duration
            last_time = time
            last_value = value
        final_duration = self.sim_hours - last_time
        if final_duration > 0:
            total_weighted_value += last_value * final_duration
        return total_weighted_value / self.sim_hours
    def calculate_all_metrics(self):
        metrics = {"tier1": {}, "tier2": {}}
        total_units = len(self.ft.revenue_events)
        summary = self.ft.get_financial_summary(self.sim_hours)
        total_mfg_cost = summary.get('totalcosts', 0)
        metrics['tier1']['manufacturing_cost_per_unit'] = {"label": "Manufacturing Cost per Unit", "value": total_mfg_cost / total_units if total_units else 0, "unit": "$/unit"}
        appraisal_cost = len(self.qt.quality_events) * self.quality_config.get('inspectioncost', 0)
        internal_failure_cost = 0
        scrap_cost = self.quality_config.get('scrapcost', 50)
        rework_cost = self.quality_config.get('reworkcost', 25)
        for event in self.qt.quality_events:
            if event.get('scrap'): internal_failure_cost += scrap_cost
            if event.get('reworked'): internal_failure_cost += rework_cost
        coq = appraisal_cost + internal_failure_cost
        metrics['tier1']['cost_of_quality'] = {"label": "Cost of Quality (CoQ)", "value": coq, "unit": "$"}
        total_revenue = summary.get('totalrevenue', 0)
        total_variable_costs = summary.get('totalprocurementcost', 0) + (summary.get('totalcosts', 0) - summary.get('totaldowntimecost', 0) - summary.get('totalprocurementcost', 0) - summary.get('totalstockoutcost',0))
        avg_margin = (total_revenue - total_variable_costs) / total_units if total_units else 0
        metrics['tier1']['unit_contribution_margin'] = {"label": "Avg. Unit Contribution Margin", "value": avg_margin, "unit": "$/unit"}
        maint_costs = sum(e.get('totalcost', 0) for e in self.ft.downtime_losses if e.get('eventtype') == 'maintenance')
        metrics['tier1']['maintenance_cost_per_unit'] = {"label": "Maintenance Cost per Unit", "value": maint_costs / total_units if total_units else 0, "unit": "$/unit"}
        metrics['tier1']['overtime_rate'] = {"label": "Overtime Rate", "value": 0, "unit": "%"}
        metrics['tier1']['overtime_cost'] = {"label": "Total Overtime Cost", "value": 0, "unit": "$"}
        cogs = total_variable_costs
        avg_inv_value = self._calculate_time_weighted_avg_inventory_value()
        metrics['tier2']['inventory_turnover_rate'] = {"label": "Inventory Turnover Rate", "value": cogs / avg_inv_value if avg_inv_value > 0 else 0, "unit": ""}
        net_income = summary.get('netprofit', 0)
        total_assets = sum(m_def.get('purchase_price', 0) for m_def in self.config.MACHINE_DEFINITIONS.values())
        metrics['tier2']['return_on_assets'] = {"label": "Return on Assets (ROA)", "value": (net_income / total_assets) * 100 if total_assets > 0 else 0, "unit": "%"}
        initial_purchase_price = total_assets
        total_downtime_cost = summary.get('totaldowntimecost', 0)
        total_energy_cost = self.et.total_energy_kwh * self.config.ENVIRONMENTCONFIG.get('gridpriceperkwh', 0.12)
        tco = initial_purchase_price + maint_costs + total_energy_cost + total_downtime_cost
        metrics['tier2']['total_cost_of_ownership'] = {"label": "Total Cost of Ownership (TCO)","value": tco, "unit": "$"}
        all_machine_stats = [machine.timeline.totals() for machine_list in self.machines.values() for machine in machine_list]
        total_possible_time = self.sim_hours * len(all_machine_stats)
        total_running_time = sum(stats.get('running', 0) for stats in all_machine_stats)
        avg_utilization = total_running_time / total_possible_time if total_possible_time else 0
        total_labor_cost = self.total_labor_cost
        unused_cost = (1 - avg_utilization) * total_labor_cost
        metrics['tier2']['unused_capacity_cost'] = {"label": "Unused Capacity Cost", "value": unused_cost, "unit": "$"}
        return metrics
    def generate_visualization_data(self):
        viz_data = {}
        summary = self.ft.get_financial_summary(self.sim_hours)
        viz_data['cost_composition_snapshot'] = [
            {"cost_component": "Downtime", "value": summary.get('totaldowntimecost', 0)},
            {"cost_component": "Procurement", "value": summary.get('totalprocurementcost', 0)},
            {"cost_component": "Stockout", "value": summary.get('totalstockoutcost', 0)},
            {"cost_component": "Labor", "value": self.total_labor_cost},
        ]
        downtime_by_machine = collections.defaultdict(float)
        for event in self.ft.downtime_losses:
            if 'machine' in event and 'Worker' not in event['machine']:
                downtime_by_machine[event['machine']] += event.get('totalcost', 0)
        pareto_data = [{"machine": m, "cost": c} for m, c in downtime_by_machine.items()]
        pareto_data.sort(key=lambda x: x['cost'], reverse=True)
        viz_data['downtime_pareto'] = pareto_data
        prod_perf = summary.get('productperformance', {})
        margin_data = []
        for prod_id, data in prod_perf.items():
            margin_data.append({
                "product": prod_id,
                "unit_contribution_margin": data.get('profitperunit', 0)
            })
        viz_data['contribution_margin_by_product'] = margin_data
        maint_costs = sum(e['totalcost'] for e in self.ft.downtime_losses if e.get('eventtype') == 'maintenance')
        failure_cost_breakdown = sum(e['totalcost'] for e in self.ft.downtime_losses if e.get('eventtype') == 'breakdown')
        scrap_cost = self.quality_config.get('scrapcost', 50)
        rework_cost = self.quality_config.get('reworkcost', 25)
        for event in self.qt.quality_events:
            if event.get('scrap'): failure_cost_breakdown += scrap_cost
            if event.get('reworked'): failure_cost_breakdown += rework_cost
        viz_data['maintenance_vs_failure_costs'] = {
            "total_maintenance_cost": maint_costs,
            "total_failure_cost": failure_cost_breakdown
        }
        return viz_data