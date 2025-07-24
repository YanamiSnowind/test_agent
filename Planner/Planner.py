from TypeEnums import TaskStatus,Task,WorkerResult
from llm import QwenLLMClient
from workers import ImageSegmentationWorker, PointCloudMappingWorker, RigidMotionWorker
from typing import Dict,List,Any,Optional
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
import json
from TypeEnums import *
import asyncio
from Planner.planner_utils.motionControl import *

class Planner:
    def __init__(self, llm_client: QwenLLMClient):
        self.llm_client = llm_client
        self.workers = {
            "image_segmentation": ImageSegmentationWorker(),
            "pointcloud_mapping": PointCloudMappingWorker(),
            "rigid_motion": RigidMotionWorker(),
            # "visualization": VisualizationWorker()
        }
        self.task_history = []
        self.knowledge_base = self._init_knowledge_base()

    async def analyze_instruction_with_thought_chain(self, instruction: str) -> List[ThoughtChain]:
        """使用思维链分析用户指令"""
        system_prompt = """
        你是一个多模态点云控制系统的分析专家。请使用思维链方法深度分析用户指令。

        分析维度：
        1. 指令理解：理解用户想要实现什么
        2. 技术路径：确定需要哪些技术步骤
        3. 资源需求：分析需要什么资源和能力
        4. 执行策略：规划如何执行
        5. 潜在风险：识别可能的问题

        返回JSON格式的思维链，每一步包含思考过程、推理逻辑、决策和置信度。
        """

        prompt = f"""
        用户指令: "{instruction}"

        请进行深度分析，返回JSON格式：
        {{
            "thought_chain": [
                {{
                    "step": 1,
                    "thought": "指令理解",
                    "reasoning": "详细的推理过程",
                    "decision": "做出的决策",
                    "confidence": 0.85
                }},
                {{
                    "step": 2,
                    "thought": "技术路径分析",
                    "reasoning": "技术实现的推理",
                    "decision": "选择的技术路径",
                    "confidence": 0.90
                }},
                {{
                    "step": 3,
                    "thought": "资源和能力评估",
                    "reasoning": "资源需求分析",
                    "decision": "资源分配策略",
                    "confidence": 0.80
                }},
                {{
                    "step": 4,
                    "thought": "执行策略制定",
                    "reasoning": "执行方案推理",
                    "decision": "具体执行策略",
                    "confidence": 0.88
                }},
                {{
                    "step": 5,
                    "thought": "风险识别",
                    "reasoning": "潜在问题分析",
                    "decision": "风险应对措施",
                    "confidence": 0.75
                }}
            ]
        }}
        """

        try:
            response = await self.llm_client.call_llm(prompt, system_prompt)
            logger.info(f"思维链分析结果: {response}")

            parsed_response = json.loads(response)
            thought_chains = []

            for chain_data in parsed_response.get("thought_chain", []):
                thought_chain = ThoughtChain(
                    step=chain_data["step"],
                    thought=chain_data["thought"],
                    reasoning=chain_data["reasoning"],
                    decision=chain_data["decision"],
                    confidence=chain_data["confidence"]
                )
                thought_chains.append(thought_chain)

            return thought_chains

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"解析思维链分析结果失败: {e}")
            return self._get_fallback_thought_chain(instruction)
    def _init_knowledge_base(self) -> Dict[str, Any]:
        """初始化领域知识库"""
        return {
            "motion_types": {
                "挥舞": {"parts": ["手", "臂"], "motion": "rotation", "axis": [0, 1, 0]},
                "摆动": {"parts": ["尾巴", "头"], "motion": "oscillation", "axis": [0, 0, 1]},
                "转动": {"parts": ["头", "身体"], "motion": "rotation", "axis": [0, 1, 0]},
                "跳跃": {"parts": ["腿", "身体"], "motion": "translation", "axis": [0, 1, 0]},
                "点头": {"parts": ["头"], "motion": "rotation", "axis": [1, 0, 0]},
                "摇头": {"parts": ["头"], "motion": "rotation", "axis": [0, 1, 0]}
            },
            "animals": {
                "松鼠": {"typical_parts": ["头", "身体", "手", "腿", "尾巴"], "size": "small"},
                "猫": {"typical_parts": ["头", "身体", "爪子", "腿", "尾巴"], "size": "medium"},
                "狗": {"typical_parts": ["头", "身体", "腿", "尾巴"], "size": "medium"},
                "人": {"typical_parts": ["头", "身体", "手", "臂", "腿"], "size": "large"}
            },
            "segmentation_strategies": {
                "small_parts": "lang_sam",  # 小部件用Lang-SAM
                "large_objects": "dino",  # 大物体用DINO
                "precise_parts": "lang_sam"  # 精确部件用Lang-SAM
            }
        }

    async def plan_tasks(self, user_instruction: str) -> List[Task]:
        system_prompt = """
            你是一个多模态点云控制系统的任务规划器。根据用户指令，你需要规划出合理的任务执行序列。

            可用的Worker类型：
            1. image_segmentation: 图像分割，识别目标区域
            2. pointcloud_mapping: 建立2D图像与3D点云的映射关系
            3. rigid_motion: 为点云添加刚性运动控制
            4. visualization: 可视化点云动画并评估效果

            请返回JSON格式的任务规划结果。        
        """
        prompt = f"""
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
        response = await self.llm_client.call_llm(prompt, system_prompt)
        logger.into(f"LLM规划结果:{response}")
        try:
            plan_data = json.loads(response)
            tasks = []
            for i, task_info in enumerate(plan_data.get("tasks", [])):
                task = Task(
                    task_id=f"task{i + 1}",
                    worker_type=task_info["worker_type"],
                    description=task_info["description"],
                    input_data=task_info.get("input_requirements", {})
                )
                tasks.append(task)
            return tasks
        except json.JSONDecodeError:
            logger.error("LLM返回的JSON格式错误")
            return self._get_default_tasks(user_instruction)

    def _get_fallback_thought_chain(self, instruction: str) -> List[ThoughtChain]:
        """获取备用思维链"""
        return [
            ThoughtChain(1, "指令理解", f"分析指令: {instruction}", "需要完整的多模态处理流程", 0.8),
            ThoughtChain(2, "技术路径分析", "需要图像分割->点云映射->运动控制->可视化", "按顺序执行四个核心步骤", 0.9),
            ThoughtChain(3, "资源评估", "需要图像处理、3D建模、动画生成能力", "使用现有worker资源", 0.85),
            ThoughtChain(4, "执行策略", "采用串行执行，每步依赖前一步结果", "顺序执行所有worker", 0.88),
            ThoughtChain(5, "风险识别", "可能存在分割不准确、映射错误等问题", "需要适当的容错机制", 0.75)
        ]
    #planner核心任务，调用llm解析指令生成制定好的方案
    async def plan_tasks_with_reasoning(self, instruction: str) -> TaskPlan:
        """基于思维链进行详细任务规划 - 主入口方法"""
        logger.info(f"开始规划任务，用户指令: {instruction}")

        # 1. 先进行思维链分析
        thought_chain = await self.analyze_instruction_with_thought_chain(instruction)
        logger.info(f"思维链分析完成，共{len(thought_chain)}个步骤")

        # 2. 基于思维链结果进行精细化任务规划
        detailed_plan = await self._create_detailed_task_plan(instruction, thought_chain)
        logger.info(f"任务规划完成，共{len(detailed_plan.tasks)}个任务")

        return detailed_plan

    async def _create_detailed_task_plan(self, instruction: str, thought_chain: List[ThoughtChain]) -> TaskPlan:
        """创建详细的任务规划"""
        logger.info("开始创建详细任务规划")

        # 1. 智能解析指令内容
        parsed_info = await self._parse_instruction_semantics(instruction)
        logger.info(f"指令语义解析完成: {parsed_info}")

        # 2. 根据解析结果和思维链生成适应性任务
        tasks = await self._generate_adaptive_tasks(instruction, parsed_info, thought_chain)
        logger.info(f"生成了{len(tasks)}个自适应任务")

        return TaskPlan(
            tasks=tasks,
            thought_chain=thought_chain
        )

    async def _parse_instruction_semantics(self, instruction: str) -> Dict[str, Any]:
        """深度解析指令语义"""
        system_prompt = """
        你是语义解析专家。请深度分析用户指令，提取关键语义信息。

        需要提取的信息：
        1. 动作主体（什么对象）
        2. 动作类型（做什么动作）  
        3. 目标部位（哪个部分）
        4. 动作特征（如何做）
        5. 技术需求（需要什么技术支持）
        """

        prompt = f"""
        指令: "{instruction}"

        请提取语义信息，返回JSON：
        {{
            "subject": "动作主体",
            "action": "动作类型", 
            "target_parts": ["目标部位列表"],
            "motion_characteristics": {{
                "type": "运动类型",
                "amplitude": "运动幅度",
                "frequency": "运动频率",
                "direction": "运动方向"
            }},
            "technical_requirements": {{
                "segmentation_precision": "分割精度要求",
                "mapping_complexity": "映射复杂度",
                "motion_complexity": "运动复杂度"
            }},
            "success_criteria": ["成功标准列表"]
        }}
        """

        response = await self.llm_client.call_llm(prompt, system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 使用知识库进行备用解析
            return self._fallback_semantic_parse(instruction)

    def _fallback_semantic_parse(self, instruction: str) -> Dict[str, Any]:
        """基于知识库的备用语义解析"""
        parsed_info = {
            "subject": "未知对象",
            "action": "未知动作",
            "target_parts": ["全身"],
            "motion_characteristics": {
                "type": "unknown",
                "amplitude": "medium",
                "frequency": "low",
                "direction": "multi"
            },
            "technical_requirements": {
                "segmentation_precision": "high",
                "mapping_complexity": "medium",
                "motion_complexity": "medium"
            },
            "success_criteria": ["动作自然", "部位准确", "效果流畅"]
        }

        # 基于知识库匹配
        for animal, info in self.knowledge_base["animals"].items():
            if animal in instruction:
                parsed_info["subject"] = animal
                parsed_info["target_parts"] = info["typical_parts"]
                break

        for motion, info in self.knowledge_base["motion_types"].items():
            if motion in instruction:
                parsed_info["action"] = motion
                parsed_info["target_parts"] = info["parts"]
                parsed_info["motion_characteristics"]["type"] = info["motion"]
                break

        return parsed_info

    async def _generate_adaptive_tasks(self, instruction: str, parsed_info: Dict, thought_chain: List[ThoughtChain]) -> \
            List[Task]:
        """生成自适应任务序列"""
        tasks = []

        # Task 1: 智能图像分割
        segmentation_strategy = self._choose_segmentation_strategy(parsed_info)
        task1 = Task(
            task_id="task_segmentation",
            worker_type="image_segmentation",
            description=f"使用{segmentation_strategy}分割{parsed_info.get('subject', '目标对象')}的{','.join(parsed_info.get('target_parts', ['目标部位']))}",
            input_data={
                "instruction": instruction,
                "target_subject": parsed_info.get("subject"),
                "target_parts": parsed_info.get("target_parts", []),
                "segmentation_method": segmentation_strategy,
                "precision_level": parsed_info.get("technical_requirements", {}).get("segmentation_precision", "high")
            }
        )
        tasks.append(task1)

        # Task 2: 自适应点云映射
        mapping_complexity = parsed_info.get("technical_requirements", {}).get("mapping_complexity", "medium")
        task2 = Task(
            task_id="task_mapping",
            worker_type="pointcloud_mapping",
            description=f"建立{parsed_info.get('subject')}的多分辨率点云映射，复杂度:{mapping_complexity}",
            input_data={
                "instruction": instruction,
                "target_parts": parsed_info.get("target_parts", []),
                "mapping_precision": "high" if mapping_complexity == "high" else "medium",
                "resolution_strategy": "multi_scale"
            }
        )
        tasks.append(task2)

        # Task 3: 精准运动控制
        motion_params = self._calculate_motion_parameters(parsed_info)
        task3 = Task(
            task_id="task_motion",
            worker_type="rigid_motion",
            description=f"为{','.join(parsed_info.get('target_parts', []))}添加{parsed_info.get('action')}运动控制",
            input_data={
                "instruction": instruction,
                "motion_type": parsed_info.get("motion_characteristics", {}).get("type", "rotation"),
                "motion_parameters": motion_params,
                "target_parts": parsed_info.get("target_parts", []),
                "animation_frames": self._calculate_frame_count(parsed_info)
            }
        )
        tasks.append(task3)

        # Task 4: 智能可视化评估
        task4 = Task(
            task_id="task_visualization",
            worker_type="visualization",
            description=f"可视化{parsed_info.get('action')}动画并智能评估效果",
            input_data={
                "instruction": instruction,
                "success_criteria": parsed_info.get("success_criteria", []),
                "evaluation_focus": parsed_info.get("target_parts", []),
                "quality_threshold": 0.8
            }
        )
        tasks.append(task4)

        return tasks

    def _choose_segmentation_strategy(self, parsed_info: Dict[str, Any]) -> str:
        """选择分割策略"""
        subject = parsed_info.get("subject", "")
        target_parts = parsed_info.get("target_parts", [])

        # 基于主体大小选择策略
        if subject in self.knowledge_base["animals"]:
            size = self.knowledge_base["animals"][subject]["size"]
            if size == "small":
                return "lang_sam"  # 小物体用精确分割
            elif size == "large":
                return "dino"  # 大物体用DINO

        # 基于部位精度要求
        if any(part in ["手", "眼", "鼻"] for part in target_parts):
            return "lang_sam"  # 精细部位用Lang-SAM

        return "dino"  # 默认使用DINO

    def _get_default_tasks(self, instruction: str) -> List[Task]:
        return [
            Task("task1", "image_segmentation", "图像分割任务", {"instruction": instruction}),
            Task("task2", "pointcloud_mapping", "点云映射任务", {"instruction": instruction}),
            Task("task3", "rigid_motion", "刚性运动任务", {"instruction": instruction}),
            Task("task4", "visualization", "可视化任务", {"instruction": instruction})
        ]
    async def execute_task_plan(self,task_plan:TaskPlan)->Optional:
        return 'success'
    async def evaluate_and_improve(self, results: Dict[str, Any]) -> bool:
        # viz_result = results.get("task4", {})
        # if viz_result.get("status") == "success":
        #     evaluation_score = viz_result.get("data", {}).get("evaluation_score", 0)
        #     feedback = viz_result.get("data", {}).get("feedback", "")
        #     logger.info(f"评估分数: {evaluation_score}, 反馈: {feedback}")
        #
        #     if evaluation_score < 0.8:
        #         logger.info("评估分数较低")
        #         return True
        return True