from abc import ABC, abstractmethod

from TypeEnums import *

class BaseWorker(ABC):
    def __init__(self,worker_id:str,worker_type:str):
        self.worker_id=worker_id
        self.worker_type=worker_type
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    @abstractmethod
    async def execute(self,task:Task)->WorkerResult:
        """通用执行"""
        pass