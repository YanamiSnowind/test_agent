from TypeEnums import TaskStatus,Task,WorkerResult
from llm import QwenLLMClient
from workers import ImageSegmentationWorker, PointCloudMappingWorker, RigidMotionWorker, VisualizationWorker
from typing import Dict,List,Any,Optional
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
import json
from TypeEnums import *
import asyncio

class Planner:
    def __init__(self, llm_client: QwenLLMClient):
        self.llm_client = llm_client
        self.workers = {
            "image_segmentation": ImageSegmentationWorker(),
            "pointcloud_mapping": PointCloudMappingWorker(),
            "rigid_motion": RigidMotionWorker(),
            "visualization": VisualizationWorker()
        }
        self.task_history = []
        self.knowledge_base = self._init_knowledge_base()

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
            ThoughtChain(1, "解析用户指令", f"分析指令: {instruction}", "需要完整的多模态处理流程", 0.8),
            ThoughtChain(2, "规划技术路径", "需要图像分割->点云映射->运动控制->可视化", "按顺序执行四个步骤", 0.9),
            ThoughtChain(3, "制定执行策略", "采用串行执行，每步依赖前一步结果", "顺序执行所有worker", 0.85),
            ThoughtChain(4, "评估风险", "可能存在分割不准确、映射错误等问题", "需要迭代优化机制", 0.7)
        ]

    async def plan_tasks_with_reasoning(self, instruction: str) -> TaskPlan:
        """基于思维链进行详细任务规划"""
        # 1. 先进行思维链分析
        thought_chain = await self.analyze_instruction_with_thought_chain(instruction)

        # 2. 基于思维链结果进行精细化任务规划
        detailed_plan = await self._create_detailed_task_plan(instruction, thought_chain)

        return detailed_plan

    async def _create_detailed_task_plan(self, instruction: str, thought_chain: List[ThoughtChain]) -> TaskPlan:
        """创建详细的任务规划"""
        # 智能解析指令内容
        parsed_info = await self._parse_instruction_semantics(instruction)

        # 根据解析结果选择合适的策略
        tasks = await self._generate_adaptive_tasks(instruction, parsed_info, thought_chain)

        # 制定执行策略
        execution_strategy = self._create_execution_strategy(parsed_info, thought_chain)

        # 风险评估
        risk_assessment = self._assess_risks(parsed_info, thought_chain)

        return TaskPlan(
            tasks=tasks,
            thought_chain=thought_chain,
            execution_strategy=execution_strategy,
            risk_assessment=risk_assessment
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

    def _choose_segmentation_strategy(self, parsed_info: Dict) -> str:
        """选择分割策略"""
        target_parts = parsed_info.get("target_parts", [])
        precision_req = parsed_info.get("technical_requirements", {}).get("segmentation_precision", "medium")

        # 小部件或高精度需求使用Lang-SAM
        small_parts = ["手", "爪子", "眼睛", "耳朵"]
        if any(part in target_parts for part in small_parts) or precision_req == "high":
            return "lang_sam"
        else:
            return "dino"

    def _calculate_motion_parameters(self, parsed_info: Dict) -> Dict[str, Any]:
        """计算运动参数"""
        motion_char = parsed_info.get("motion_characteristics", {})
        action = parsed_info.get("action", "")

        # 从知识库获取基础参数
        base_params = self.knowledge_base["motion_types"].get(action, {
            "motion": "rotation",
            "axis": [0, 1, 0]
        })

        return {
            "rotation_axis": base_params.get("axis", [0, 1, 0]),
            "rotation_angle": self._map_amplitude_to_angle(motion_char.get("amplitude", "medium")),
            "translation": [0, 0, 0],  # 大多数情况下不需要平移
            "motion_type": base_params.get("motion", "rotation"),
            "easing_function": "ease_in_out"
        }

    def _map_amplitude_to_angle(self, amplitude: str) -> float:
        """将幅度描述映射为角度"""
        amplitude_map = {
            "small": 15.0,
            "medium": 30.0,
            "large": 60.0,
            "extreme": 90.0
        }
        return amplitude_map.get(amplitude, 30.0)

    def _calculate_frame_count(self, parsed_info: Dict) -> int:
        """计算动画帧数"""
        frequency = parsed_info.get("motion_characteristics", {}).get("frequency", "low")
        frequency_map = {
            "low": 20,
            "medium": 30,
            "high": 60
        }
        return frequency_map.get(frequency, 30)

    def _create_execution_strategy(self, parsed_info: Dict, thought_chain: List[ThoughtChain]) -> Dict[str, Any]:
        """制定执行策略"""
        return {
            "execution_mode": "sequential",  # 串行执行
            "retry_policy": {
                "max_retries": 2,
                "retry_conditions": ["low_confidence", "evaluation_failed"]
            },
            "optimization_strategy": {
                "enable_iterative_improvement": True,
                "max_iterations": 3,
                "improvement_threshold": 0.8
            },
            "checkpoints": {
                "save_intermediate_results": True,
                "checkpoint_tasks": ["task_segmentation", "task_mapping"]
            }
        }

    def _assess_risks(self, parsed_info: Dict, thought_chain: List[ThoughtChain]) -> Dict[str, Any]:
        """评估执行风险"""
        risks = {
            "segmentation_risks": {
                "small_parts_detection": 0.3 if any(
                    "手" in part or "爪" in part for part in parsed_info.get("target_parts", [])) else 0.1,
                "occlusion_handling": 0.2,
                "boundary_accuracy": 0.15
            },
            "mapping_risks": {
                "resolution_mismatch": 0.25,
                "correspondence_errors": 0.3,
                "depth_ambiguity": 0.2
            },
            "motion_risks": {
                "unnatural_movement": 0.2,
                "part_collision": 0.15,
                "motion_continuity": 0.1
            },
            "overall_confidence": sum(tc.confidence for tc in thought_chain) / len(
                thought_chain) if thought_chain else 0.5
        }

        return risks

    async def plan_tasks(self, user_instruction: str) -> List[Task]:
        """主要的任务规划接口（保持兼容性）"""
        task_plan = await self.plan_tasks_with_reasoning(user_instruction)
        return task_plan.tasks

    async def execute_tasks_with_monitoring(self, task_plan: TaskPlan) -> Dict[str, Any]:
        """带监控的任务执行"""
        tasks = task_plan.tasks
        execution_strategy = task_plan.execution_strategy
        risk_assessment = task_plan.risk_assessment

        results = {}
        execution_context = {
            "current_step": 0,
            "total_steps": len(tasks),
            "intermediate_results": {},
            "performance_metrics": {}
        }

        logger.info("=" * 60)
        logger.info("开始智能任务执行流程")
        logger.info(f"总任务数: {len(tasks)}")
        logger.info(f"整体风险评估置信度: {risk_assessment.get('overall_confidence', 0):.2f}")
        logger.info("=" * 60)

        for i, task in enumerate(tasks):
            execution_context["current_step"] = i + 1

            logger.info(f"\n[步骤 {i + 1}/{len(tasks)}] 执行任务: {task.task_id}")
            logger.info(f"任务描述: {task.description}")

            # 执行前的智能检查
            pre_check_result = await self._pre_execution_check(task, execution_context, risk_assessment)
            if not pre_check_result["proceed"]:
                logger.warning(f"任务 {task.task_id} 预检查未通过: {pre_check_result['reason']}")
                task.status = TaskStatus.FAILED
                task.error_msg = pre_check_result["reason"]
                continue

            # 执行任务
            task.status = TaskStatus.RUNNING
            worker = self.workers.get(task.worker_type)

            if not worker:
                task.status = TaskStatus.FAILED
                task.error_msg = f"未找到对应的Worker: {task.worker_type}"
                results[task.task_id] = {"status": "failed", "error": task.error_msg}
                continue

            try:
                # 智能数据传递 - 根据任务类型选择相关数据
                enhanced_input = await self._prepare_enhanced_input(task, execution_context)
                task.input_data.update(enhanced_input)

                # 执行任务
                start_time = asyncio.get_event_loop().time()
                result = await worker.execute(task)
                execution_time = asyncio.get_event_loop().time() - start_time

                # 记录性能指标
                execution_context["performance_metrics"][task.task_id] = {
                    "execution_time": execution_time,
                    "success": result.success,
                    "confidence": getattr(result, 'confidence', 0.0)
                }

                if result.success:
                    task.status = TaskStatus.COMPLETED
                    task.output_data = result.data

                    # 执行后的智能验证
                    validation_result = await self._post_execution_validation(task, result, risk_assessment)

                    if validation_result["valid"]:
                        results[task.task_id] = {"status": "success", "data": result.data}
                        execution_context["intermediate_results"][task.task_id] = result.data
                        logger.info(f"✓ 任务 {task.task_id} 执行成功，耗时 {execution_time:.2f}s")
                    else:
                        logger.warning(f"⚠ 任务 {task.task_id} 结果验证失败: {validation_result['reason']}")
                        # 根据策略决定是否重试
                        if self._should_retry(task, execution_strategy):
                            logger.info(f"准备重试任务 {task.task_id}")
                            # 重试逻辑 (简化版，可扩展)
                            retry_result = await worker.execute(task)
                            if retry_result.success:
                                task.status = TaskStatus.COMPLETED
                                results[task.task_id] = {"status": "success", "data": retry_result.data}
                            else:
                                task.status = TaskStatus.FAILED
                                results[task.task_id] = {"status": "failed", "error": retry_result.error}
                        else:
                            task.status = TaskStatus.FAILED
                            results[task.task_id] = {"status": "failed", "error": validation_result["reason"]}
                else:
                    task.status = TaskStatus.FAILED
                    task.error_msg = result.error
                    results[task.task_id] = {"status": "failed", "error": result.error}
                    logger.error(f"✗ 任务 {task.task_id} 执行失败: {result.error}")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_msg = str(e)
                results[task.task_id] = {"status": "failed", "error": str(e)}
                logger.error(f"✗ 任务 {task.task_id} 执行异常: {e}")

        # 添加执行总结
        results["execution_summary"] = {
            "total_tasks": len(tasks),
            "successful_tasks": sum(
                1 for r in results.values() if isinstance(r, dict) and r.get("status") == "success"),
            "performance_metrics": execution_context["performance_metrics"],
            "thought_chain_summary": [
                {"step": tc.step, "thought": tc.thought, "confidence": tc.confidence}
                for tc in task_plan.thought_chain
            ]
        }

        self.task_history.extend(tasks)
        return results

    async def _pre_execution_check(self, task: Task, context: Dict, risk_assessment: Dict) -> Dict[str, Any]:
        if task.work_type != "image_segmentation" and context["current_step"] > 1:
            prev_results = context.get("intermediate_results", {})
            if not prev_results:
                return {"proceed": False, "reason": "缺少任务"}
        task_risks = risk_assessment.get(f"{task.work_type}_risks", {})
        max_risk = max(task_risks.values()) if task_risks else 0
        if max_risk > 0.5:
            logger.warning(f"任务 {task.task_id} 风险等级较高: {max_risk:.2f}")
        return {"proceed": True, "reason": "预检查通过"}

    async def _prepare_enhanced_input(self, task: Task, context: Dict) -> Dict[str, Any]:
        """准备增强的输入数据"""
        enhanced_input = {}

        # 从上下文中获取相关的中间结果
        intermediate_results = context.get("intermediate_results", {})

        # if task.worker_type == "pointcloud_mapping":
        #     # 点云映射任务需要分割结果
        #     seg_result = intermediate_results.get("task_segmentation", {
        #
        #     async
        #
        #     def evaluate_and_improve(self, results: Dict[str, Any]) -> bool:
        #
        #         """评估结果并决定是否需要改进"""
        #     # 获取可视化worker的评估结果
        #     viz_result = results.get("task4", {})
        #     if viz_result.get("status") == "success":
        #         evaluation_score = viz_result.get("data", {}).get("evaluation_score", 0)
        #         feedback = viz_result.get("data", {}).get("feedback", "")
        #
        #         logger.info(f"评估分数: {evaluation_score}, 反馈: {feedback}")
        #
        #         # 如果评估分数低于阈值，需要改进
        #         if evaluation_score < 0.8:
        #             logger.info("评估分数较低，需要重新调整映射网络")
        #             return True
        #
        #     return False

    def _get_default_tasks(self, instruction: str) -> List[Task]:
        return [
            Task("task1", "image_segmentation", "图像分割任务", {"instruction": instruction}),
            Task("task2", "pointcloud_mapping", "点云映射任务", {"instruction": instruction}),
            Task("task3", "rigid_motion", "刚性运动任务", {"instruction": instruction}),
            Task("task4", "visualization", "可视化任务", {"instruction": instruction})
        ]

    async def evaluate_and_improve(self, results: Dict[str, Any]) -> bool:
        viz_result = results.get("task4", {})
        if viz_result.get("status") == "success":
            evaluation_score = viz_result.get("data", {}).get("evaluation_score", 0)
            feedback = viz_result.get("data", {}).get("feedback", "")
            logger.info(f"评估分数: {evaluation_score}, 反馈: {feedback}")

            if evaluation_score < 0.8:
                logger.info("评估分数较低")
                return True
        return False