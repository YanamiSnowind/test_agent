"""
基于Agent控制的系统
"""
import json
import asyncio
import logging
from typing import Dict,List,Any,Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC,abstractmethod
import openai

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)

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

class QwenLLMClient:
    def __init__(self,api_key="",base_url=""):
        self.client=openai.OpenAI(api_key,base_url)
    async def call_llm(self,prompt:str,system_prompt:str="")->str:
        try:
            messages=[]
            if system_prompt:
                messages.append({"role":"system","content":system_prompt})
            messages.append({"role":"user","content":prompt})

            response=self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.7
            )
            return response.choice[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败{e}")
            return f"Error{e}"

class BaseWorker(ABC):
    def __init__(self,worker_id:str,worker_type:str):
        self.worker_id=worker_id
        self.worker_type=worker_type
    @abstractmethod
    async def execute(self,task:Task)->WorkerResult:
        """通用执行"""
        pass
class ImageSegmentationWorker(BaseWorker):
    def __init__(self):
        super(ImageSegmentationWorker, self).__init__("worker1","image_segmentation")
    async def execute(self,task:Task) ->WorkerResult:
        logger.info(f"Worker1执行图像分割任务:{task.description}")
        await asyncio.sleep(1)
        #模拟一个分割后的结果
        mock_result={
            "segmented_regions":["region_1","region_2"],
            "pixel_coordinates":[[100,200],[300,400]],
            "confidence_scores":[0.95,0.87]
        }
        return WorkerResult(success=True,data=mock_result)

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


class VisualizationWorker(BaseWorker):
    """Worker4: 可视化工作器"""

    def __init__(self):
        super().__init__("worker4", "visualization")

    async def execute(self, task: Task) -> WorkerResult:
        """执行可视化任务"""
        logger.info(f"Worker4执行可视化任务: {task.description}")

        await asyncio.sleep(3)

        # TODO: 实际调用pcdviewer并截屏
        mock_result = {
            "screenshots": [f"screenshot_{i}.png" for i in range(10)],
            "video_path": "motion_animation.mp4",
            "evaluation_score": 0.85,
            "feedback": "动作效果良好，建议调整旋转角度"
        }

        return WorkerResult(success=True, data=mock_result)

class Planner:
    def __init__(self,llm_client:QwenLLMClient):
        self.llm_client=llm_client
        self.workers={
            "image_segmentation":ImageSegmentationWorker(),
            "pointcloud_mapping":PointCloudMappingWorker(),
            "rigid_motion":RigidMotionWorker(),
            "visualization":VisualizationWorker()
        }
        self.task_history=[]
    async def plan_tasks(self,user_instruction:str)->List[Task]:
        system_prompt="""
            你是一个多模态点云控制系统的任务规划器。根据用户指令，你需要规划出合理的任务执行序列。
            
            可用的Worker类型：
            1. image_segmentation: 图像分割，识别目标区域
            2. pointcloud_mapping: 建立2D图像与3D点云的映射关系
            3. rigid_motion: 为点云添加刚性运动控制
            4. visualization: 可视化点云动画并评估效果
            
            请返回JSON格式的任务规划结果。        
        """
        prompt=f"""
        用户指令: {user_instruction}
        请规划执行任务序列，返回JSON格式：
        {{
            "tasks": [
                {{
                    "worker_type": "worker类型",
                    "description": "任务描述",
                    "input_requirements": {{"key": "value"}}
                }}
            ],
            "execution_order": ["task1", "task2", "task3", "task4"],
            "dependencies": {{"task2": ["task1"], "task3": ["task2"]}}
        }}        
        """
        response=await self.llm_client.call_llm(prompt,system_prompt)
        logger.into(f"LLM规划结果:{response}")
        try:
            plan_data=json.loads(response)
            tasks=[]
            for i,task_info in enumerate(plan_data.get("tasks",[])):
                task=Task(
                    task_id=f"task{i+1}",
                    worker_type=task_info["worker_type"],
                    description=task_info["description"],
                    input_data=task_info.get("input_requirements",{})
                )
                tasks.append(task)
            return tasks
        except json.JSONDecodeError:
            logger.error("LLM返回的JSON格式错误")
            return self._get_default_tasks(user_instruction)
    def _get_default_tasks(self,instruction:str)->List[Task]:
        return[
            Task("task1","image_segmentation","图像分割任务",{"instruction":instruction}),
            Task("task2", "pointcloud_mapping", "点云映射任务", {"instruction": instruction}),
            Task("task3", "rigid_motion", "刚性运动任务", {"instruction": instruction}),
            Task("task4", "visualization", "可视化任务", {"instruction": instruction})
        ]
    async def evaluate_and_improve(self,results:Dict[str,Any])->bool:
        viz_result=results.get("task4",{})
        if viz_result.get("status")=="success":
            evaluation_score=viz_result.get("data",{}).get("evaluation_score",0)
            feedback=viz_result.get("data",{}).get("feedback","")
            logger.info(f"评估分数: {evaluation_score}, 反馈: {feedback}")

            if evaluation_score<0.8:
                logger.info("评估分数较低")
                return True
        return False

class AgentSystem:
    def __init__(self,llm_api_key:str="",llm_base_url:str=""):
        self.llm_client=QwenLLMClient(llm_api_key,llm_base_url)
        self.planner=Planner(self.llm_client)
        self.max_iterations=3
    async def process_user_instruction(self,instruction:str)->str:
        logger.info(f"接收用户指令:{instruction}")
        iteration=0
        while iteration<self.max_iterations:
            iteration+=1
            logger.info(f"开始第{iteration}次迭代执行")

            tasks=await self.planner.plan_tasks(instruction)

            results=await self.planner.execute_tasks(tasks)

            need_improvement=await self.planner.evaluate_and_improve(results)

            if not need_improvement:
                logger.info("任务执行成功，效果满意")
                return self.generate_completion_message(results)
            else:
                logger.info(f"第{iteration}次迭代结果不满意，准备重新执行")
                if iteration<self.max_iterations:
                    continue
                else:
                    logger.warning("达到最大迭代次数")
                    return self._generate_completion_message(results)
        return "任务执行完成"

    def _generate_completion_message(self, results: Dict[str, Any]) -> str:
        """生成任务完成消息"""
        successful_tasks = sum(1 for r in results.values() if r.get("status") == "success")
        total_tasks = len(results)

        message = f"任务执行完成！成功完成 {successful_tasks}/{total_tasks} 个子任务。\n"

        # 添加关键结果信息
        if "task4" in results and results["task4"].get("status") == "success":
            viz_data = results["task4"].get("data", {})
            if "evaluation_score" in viz_data:
                message += f"动画效果评估分数: {viz_data['evaluation_score']:.2f}\n"
            if "screenshots" in viz_data:
                message += f"生成截图数量: {len(viz_data['screenshots'])}\n"

        return message


async def main():
    """主函数"""
    print("=" * 60)
    print("多模态点云控制Agent系统")
    print("=" * 60)

    # 初始化系统
    # 注意：请替换为实际的API密钥和地址
    system = AgentSystem(
        llm_api_key="your_actual_api_key",
        llm_base_url="your_actual_base_url"
    )

    while True:
        try:
            # 获取用户输入
            instruction = input("\n请输入控制指令 (输入'quit'退出): ").strip()

            if instruction.lower() in ['quit', 'exit', 'q']:
                print("系统退出")
                break

            if not instruction:
                print("请输入有效的指令")
                continue

            # 处理指令
            print("\n正在处理您的指令，请稍候...")
            completion_message = await system.process_user_instruction(instruction)

            print("\n" + "=" * 50)
            print("任务完成报告:")
            print("=" * 50)
            print(completion_message)

        except KeyboardInterrupt:
            print("\n\n系统被用户中断")
            break
        except Exception as e:
            print(f"系统错误: {e}")
            logger.error(f"系统错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())