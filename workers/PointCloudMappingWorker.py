from workers import BaseWorker
from TypeEnums import *
import asyncio

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
class PointCloudMappingWorker(BaseWorker):
    def __init__(self):
        super(PointCloudMappingWorker, self).__init__("worker2","pointcloud_segmentation")
    async def execute(self,task)->WorkerResult:
        logger.info(f"Worker2执行3M映射任务:{task.description}")
        await asyncio.sleep(2)
        mock_data={
            "mapped_pointcloud":"mapped_points.pcd",
            "pointcloud_coordinates":[[100,200,300],[100,200,300]],
            "target_points":[[x,y,z] for x,y,z in[(1,2,3),(4,5,6)]],
            "confidence":0.92
        }
        return WorkerResult(success=True,data=mock_data)