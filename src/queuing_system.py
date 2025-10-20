import simpy
from dataclasses import dataclass, field
from typing import Any, List, Dict, Callable
from dispatch_policies import get_dispatch_policy
import math
def mm1_model(lam, mu, **kwargs):
    if lam >= mu:
        return {"error": "Arrival rate must be less than service rate (λ < μ)"}
    rho = lam / mu  
    Lq = (lam ** 2) / (mu * (mu - lam))  
    Wq = Lq / lam  
    W = Wq + (1 / mu)  
    L = lam * W  
    return {"rho": rho, "Lq": Lq, "Wq": Wq, "L": L, "W": W, "model": "M/M/1"}
def mmc_model(lam, mu, c, **kwargs):
    if lam >= c * mu:
        return {"error": "Arrival rate must be less than total service capacity (λ < c*μ)"}
    rho = lam / (c * mu)
    term1 = ((c * rho) ** c) / math.factorial(c)
    sum_term = sum([((c * rho) ** i) / math.factorial(i) for i in range(c)])
    C_c = term1 / (term1 + (1 - rho) * sum_term)
    Lq = C_c * (rho / (1 - rho))
    Wq = Lq / lam
    W = Wq + (1 / mu)
    L = lam * W
    return {"rho": rho, "Lq": Lq, "Wq": Wq, "L": L, "W": W, "C_c": C_c, "model": f"M/M/{c}"}
def mg1_model(lam, mu, service_scv, **kwargs):
    if lam >= mu:
        return {"error": "Arrival rate must be less than service rate (λ < μ)"}
    rho = lam / mu
    Lq = (lam**2 * (1/mu)**2 * (1 + service_scv)) / (2 * (1 - rho))
    Wq = Lq / lam
    W = Wq + (1 / mu)
    L = Lq + rho
    return {"rho": rho, "Lq": Lq, "Wq": Wq, "L": L, "W": W, "model": "M/G/1"}
def gg1_kingman_approx(lam: float, mu: float, arrival_scv: float, service_scv: float) -> dict:
    if lam >= mu:
        return {"error": "Arrival rate must be less than service rate (λ < μ)"}
    rho = lam / mu
    Wq = (1/mu) * (rho / (1 - rho)) * ((arrival_scv + service_scv) / 2)
    Lq = lam * Wq
    W = Wq + (1 / mu)
    L = Lq + rho
    return {"rho": rho, "Lq": Lq, "Wq": Wq, "L": L, "W": W, "model": "G/G/1 (Kingman)"}
ANALYTICAL_MODELS = {
    "M/M/1": mm1_model,
    "M/M/c": mmc_model,
    "M/G/1": mg1_model,
    "G/G/1": gg1_kingman_approx,
}
@dataclass
class Job:
    job_id: Any
    product_type: str
    stage: str
    arrival_time: float
    process: Any 
    due_date: float = float('inf')
    profit: float = 0.0
    processing_time_estimate: float = 0.0
    request_event: simpy.Event = field(default=None, repr=False)
    routing: List[str] = field(default_factory=list, repr=False)
    context: Dict = field(default_factory=dict, repr=False)
    next_machine: 'EnhancedMachine' = None 
class BaseQueuePolicy:
    def select_job(self, jobs: List[Job], current_time: float) -> Job:
        raise NotImplementedError
class FIFO(BaseQueuePolicy):
    def select_job(self, jobs: List[Job], current_time: float) -> Job:
        return min(jobs, key=lambda j: j.arrival_time)
class LIFO(BaseQueuePolicy):
    def select_job(self, jobs: List[Job], current_time: float) -> Job:
        return max(jobs, key=lambda j: j.arrival_time)
class SJF(BaseQueuePolicy):
    def select_job(self, jobs: List[Job], current_time: float) -> Job:
        return min(jobs, key=lambda j: j.processing_time_estimate)
class Priority(BaseQueuePolicy):
    def __init__(self, criteria: List[Dict]):
        self.criteria = criteria
    def select_job(self, jobs: List[Job], current_time: float) -> Job:
        sorted_jobs = list(jobs)
        for c in reversed(self.criteria):
            attribute = c.get("attribute")
            ascending = c.get("ascending", True)
            if jobs and hasattr(jobs[0], attribute):
                sorted_jobs.sort(key=lambda j: getattr(j, attribute), reverse=not ascending)
        return sorted_jobs[0]
from dispatch_policies import SPTPolicy, EDDPolicy, WSPTPolicy, ATCPolicy
_ALL_POLICIES = {
    'FIFO': FIFO,
    'LIFO': LIFO,
    'SJF': SJF,
    'PRIORITY': Priority,
    'SPT': SPTPolicy,
    'EDD': EDDPolicy,
    'WSPT': WSPTPolicy,
    'ATC': ATCPolicy,
}
def get_policy_from_config(policy_config: dict) -> BaseQueuePolicy:
    policy_name = policy_config.get("policy", "FIFO").upper()
    params = policy_config.get("params", {})
    policy_class = _ALL_POLICIES.get(policy_name)
    if not policy_class:
        print(f"FATAL ERROR: Policy '{policy_name}' is not defined in the _ALL_POLICIES map. Defaulting to FIFO.")
        return FIFO()
    if policy_name == 'PRIORITY':
        criteria = policy_config.get("criteria", [])
        return Priority(criteria)
    try:
        return policy_class(**params)
    except TypeError:
        print(f"WARNING: Policy '{policy_name}' was called with unused params {params}. Instantiating without them.")
        return policy_class()
class PolicyQueue:
    def __init__(self, env: simpy.Environment, policy: BaseQueuePolicy, machine_name: str, metrics_bus: Any = None, capacity: int = float('inf')):
        self.env = env
        self.policy = policy
        self.machine_name = machine_name
        self.metrics_bus = metrics_bus
        self.jobs: List[Job] = []
        self.capacity = capacity
        self.put_event = env.event() 
        self.get_event = env.event() 
    def enqueue(self, job: Job, prepend: bool = False):
        if prepend:
            self.jobs.insert(0, job)
        else:
            self.jobs.append(job)
        if self.metrics_bus:
            self.metrics_bus.observe(f"queue_length.{self.machine_name}", len(self.jobs), self.env.now)
        if not self.put_event.triggered:
            self.put_event.succeed()
            self.put_event = self.env.event()
    def dequeue(self) -> Job:
        if not self.jobs:
            yield self.put_event
        selected_job = self.policy.select_job(self.jobs, self.env.now)
        self.jobs.remove(selected_job)
        print(f"--> Time {self.env.now:.2f}: DEQUEUE on '{self.machine_name}'. Job '{selected_job.job_id}' selected by {self.policy.__class__.__name__} policy.")
        if not self.get_event.triggered:
            self.get_event.succeed()
            self.get_event = self.env.event()
        if self.metrics_bus:
            wait_time = self.env.now - selected_job.arrival_time
            self.metrics_bus.observe(f"queue_wait_time.{self.machine_name}", wait_time, self.env.now)
            self.metrics_bus.observe(f"queue_length.{self.machine_name}", len(self.jobs), self.env.now)
        return selected_job
    def __len__(self):
        return len(self.jobs)