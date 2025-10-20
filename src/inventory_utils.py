from typing import Dict
import math
def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    annual_holding_cost_per_unit: float
) -> float:
    if annual_demand <= 0 or ordering_cost < 0 or annual_holding_cost_per_unit <= 0:
        return 0
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost_per_unit)
    return math.ceil(eoq)
def abc_classification(
    item_consumption_values: Dict[str, float],
    a_threshold: float = 0.80,
    b_threshold: float = 0.95
) -> Dict[str, str]:
    if not item_consumption_values:
        return {}
    total_value = sum(item_consumption_values.values())
    if total_value == 0:
        return {item: 'C' for item in item_consumption_values}
    sorted_items = sorted(
        item_consumption_values.items(),
        key=lambda item: item[1],
        reverse=True
    )
    classification = {}
    cumulative_value = 0.0
    for item_name, value in sorted_items:
        cumulative_value += value
        cumulative_percentage = cumulative_value / total_value
        if cumulative_percentage <= a_threshold:
            classification[item_name] = 'A'
        elif cumulative_percentage <= b_threshold:
            classification[item_name] = 'B'
        else:
            classification[item_name] = 'C'
    return classification