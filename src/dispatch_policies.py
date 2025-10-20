import math
from typing import List, Optional
class Job:
    def __init__(self, jobid, producttype, stage, arrivaltime, processing_time_estimate, due_date, profit=0.0):
        self.jobid = jobid
        self.producttype = producttype
        self.stage = stage
        self.arrivaltime = arrivaltime
        self.processing_time_estimate = processing_time_estimate
        self.due_date = due_date
        self.profit = profit  
class DispatchPolicyBase:
    def select_job(self, jobs: List[Job], current_time: float) -> Optional[Job]:
        raise NotImplementedError("Each policy must implement the 'select_job' method.")
    def __str__(self):
        return self.__class__.__name__
class SPTPolicy(DispatchPolicyBase):
    def select_job(self, jobs: List[Job], current_time: float) -> Optional[Job]:
        if not jobs:
            return None
        return min(jobs, key=lambda j: j.processing_time_estimate)
class EDDPolicy(DispatchPolicyBase):
    def select_job(self, jobs: List[Job], current_time: float) -> Optional[Job]:
        if not jobs:
            return None
        return min(jobs, key=lambda j: j.due_date)
class WSPTPolicy(DispatchPolicyBase):
    def select_job(self, jobs: List[Job], current_time: float) -> Optional[Job]:
        if not jobs:
            return None
        eligible_jobs = [j for j in jobs if j.processing_time_estimate > 0]
        if not eligible_jobs:
            return min(jobs, key=lambda j: j.processing_time_estimate) if jobs else None
        return max(eligible_jobs, key=lambda j: (j.profit if j.profit > 0 else 1.0) / j.processing_time_estimate)
class ATCPolicy(DispatchPolicyBase):
    def __init__(self, k: float = 2.0):
        self.k = k
    def select_job(self, jobs: List[Job], current_time: float) -> Optional[Job]:
        if not jobs:
            return None
        avg_processing_time = sum(j.processing_time_estimate for j in jobs) / len(jobs)
        if avg_processing_time <= 0:
            avg_processing_time = 1 
        def atc_index(job: Job) -> float:
            weight = job.profit if job.profit > 0 else 1.0
            processing_time = job.processing_time_estimate if job.processing_time_estimate > 0 else 1.0
            slack_time = job.due_date - current_time - processing_time
            urgency_factor = math.exp(-max(slack_time, 0) / (self.k * avg_processing_time))
            return (weight / processing_time) * urgency_factor
        return max(jobs, key=atc_index)
_policy_map = {
    'SPT': SPTPolicy,
    'EDD': EDDPolicy,
    'WSPT': WSPTPolicy,
    'ATC': ATCPolicy
}
def get_dispatch_policy(policy_name: str, **kwargs) -> DispatchPolicyBase:
    policy_class = _policy_map.get(policy_name.upper())
    if not policy_class:
        print(f"Warning: Dispatch policy '{policy_name}' not recognized. Falling back to SPT.")
        return SPTPolicy()
    return policy_class(**kwargs)