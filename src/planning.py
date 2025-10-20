from typing import List, Dict, Sequence, Tuple
import math
import statistics
def get_canonical_stages(products_config: Dict[str, Dict]) -> List[str]:
    stage_positions = {}
    for prod_conf in products_config.values():
        routing = prod_conf.get('routing', [])
        for i, stage in enumerate(routing):
            if stage not in stage_positions:
                stage_positions[stage] = []
            stage_positions[stage].append(i)
    avg_positions = {
        stage: statistics.mean(positions)
        for stage, positions in stage_positions.items()
    }
    sorted_stages = sorted(avg_positions.keys(), key=lambda stage: avg_positions[stage])
    return sorted_stages
def _prepare_processing_times(
    jobs: List[Dict],
    products_config: Dict[str, Dict],
    canonical_stages: Sequence[str]
) -> Dict[str, List[float]]:
    processing_matrix = {}
    for job in jobs:
        job_id = job['id']
        product_type = job['product_type']
        if product_type not in products_config:
            print(f"WARNING (Planning): Product type '{product_type}' for job '{job_id}' not found in PRODUCTS config. Skipping.")
            continue
        prod_conf = products_config[product_type]
        job_times = []
        for stage in canonical_stages:
            time_range = prod_conf.get('machine_times', {}).get(stage)
            if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                processing_time = (time_range[0] + time_range[1]) / 2.0
            elif isinstance(time_range, (int, float)):
                processing_time = float(time_range)
            else:
                processing_time = 0.0
            job_times.append(processing_time)
        processing_matrix[job_id] = job_times
    return processing_matrix
def _palmer_heuristic(
    processing_matrix: Dict[str, List[float]],
    stages: Sequence[str]
) -> List[str]:
    num_machines = len(stages)
    job_priorities = []
    for job_id, times in processing_matrix.items():
        priority = 0
        for k, time in enumerate(times, start=1):
            weight = (2 * k) - (num_machines + 1)
            priority += weight * time
        total_time = sum(times)
        job_priorities.append((priority, total_time, job_id))
    job_priorities.sort()
    return [job_id for priority, total_time, job_id in job_priorities]
def _cds_heuristic(
    processing_matrix: Dict[str, List[float]],
    stages: Sequence[str]
) -> List[str]:
    num_machines = len(stages)
    if num_machines <= 1:
        return list(processing_matrix.keys())
    best_sequence = []
    min_makespan = float('inf')
    job_ids = list(processing_matrix.keys())
    for k in range(1, num_machines):
        two_machine_times = []
        for job_id in job_ids:
            p_times = processing_matrix[job_id]
            time_m1 = sum(p_times[:k])
            time_m2 = sum(p_times[k:])
            two_machine_times.append((job_id, time_m1, time_m2))
        left, right = [], []
        while two_machine_times:
            min_time = float('inf')
            min_job_idx = -1
            is_on_m1 = False
            for i, (job_id, t1, t2) in enumerate(two_machine_times):
                if t1 < min_time:
                    min_time, min_job_idx, is_on_m1 = t1, i, True
                if t2 < min_time:
                    min_time, min_job_idx, is_on_m1 = t2, i, False
            job_id, t1, t2 = two_machine_times.pop(min_job_idx)
            if is_on_m1:
                left.append(job_id)
            else:
                right.insert(0, job_id)
        current_sequence = left + right
        completion_times = [[0] * num_machines for _ in range(len(current_sequence))]
        for i, job_id in enumerate(current_sequence):
            for j in range(num_machines):
                up = completion_times[i - 1][j] if i > 0 else 0
                left = completion_times[i][j - 1] if j > 0 else 0
                completion_times[i][j] = max(up, left) + processing_matrix[job_id][j]
        makespan = completion_times[-1][-1]
        if makespan < min_makespan:
            min_makespan = makespan
            best_sequence = current_sequence
    return best_sequence
def create_master_sequence(
    jobs: List[Dict],
    products_config: Dict[str, Dict],
    method: str = 'palmer'
) -> List[str]:
    if not jobs:
        return []
    canonical_stages = get_canonical_stages(products_config)
    processing_matrix = _prepare_processing_times(jobs, products_config, canonical_stages)
    method = (method or 'palmer').lower()
    if method == 'palmer':
        return _palmer_heuristic(processing_matrix, canonical_stages)
    elif method == 'cds':
        return _cds_heuristic(processing_matrix, canonical_stages)
    else:
        raise ValueError(f"Unknown planning method '{method}'. Use 'palmer' or 'cds'.")