from dataclasses import dataclass
from enum import Enum
from typing import Dict,List,Any,Optional

class TaskStatus(Enum):
    PENDING="pending"
    RUNNING="running"
    COMPLETED="completed"
    FAILED="failed"


@dataclass
class Task:
    task_id:str
    work_type:str
    description:str
    input_data:Dict[str,Any]
    output_data:Optional[Dict[str,Any]]=None
    status:TaskStatus=TaskStatus.PENDING
    error_msg:Optional[str]=None

@dataclass
class WorkerResult:
    success:bool
    data:Optional[Dict[str,Any]]=None
    error:Optional[str]=None


@dataclass
class ThoughtChain:
    step:int
    thought:str
    reasoning:str
    decision:str
    confidence:float

@dataclass
class TaskPlan:
    tasks:List[Task]
    thought_chain:List[ThoughtChain]
    execution_strategy:Dict[str,Any]
    risk_assessment:Dict[str,Any]