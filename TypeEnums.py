from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
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
    #execution_strategy:Dict[str,Any]
    # risk_assessment:Dict[str,Any]

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
class BaseWorker(ABC):
    def __init__(self,worker_id:str,worker_type:str):
        self.worker_id=worker_id
        self.worker_type=worker_type
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    @abstractmethod
    async def execute(self,task:Task)->WorkerResult:
        """通用执行"""
        pass