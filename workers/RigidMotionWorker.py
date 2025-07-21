from workers import BaseWorker
from TypeEnums import *
import asyncio
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
class RigidMotionWorker(BaseWorker):
    def __init__(self):
        super(RigidMotionWorker, self).__init__("worker3","motionControl")
    async def execute(self,task)->WorkerResult:
        logger.info(f"Worker3执行移动控制任务:{task.description}")
        await asyncio.sleep(3)
        mock_data={
            "motion_frames":[f"frame_{i}.pcd" for i in range(10)],
            "motion_parameters":{
                "rotation_axis":[0,1,0],
                "rotation_angle":30,
                "translation":[0,0,0]
            },
            "frame_count":10
        }
        return WorkerResult(success=True,data=mock_data)