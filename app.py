import json
import asyncio
import logging
from typing import Dict,List,Any,Optional
from workers import ImageSegmentationWorker, PointCloudMappingWorker, RigidMotionWorker
# from llm import QwenLLMClient
from Planner import Planner
from dataclasses import dataclass
from enum import Enum
from abc import ABC,abstractmethod
from workers import *
from TypeEnums import *
import openai
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

class ImageSegmentationWorker(BaseWorker):
    def __init__(self):
        super(ImageSegmentationWorker, self).__init__("worker1","image_segmentation")
        self.supported_methods={
            "lang_sam":self._execute_lang_sam,
            "dino":self._execute_dino,
        }
        self.min_quality_threshold=0.7
        self.max_retry_attempts=3
    async def execute(self,task:Task) ->WorkerResult:
        try:
            logger.info(f"Worker1开始执行:{task.description}")
            instruction=task.input_data.get("instruction","")
            target_subject=task.input_data.get("target_subject",[])
            target_parts = task.input_data.get("target_parts", [])
            segmentation_method = task.input_data.get("segmentation_method", "lang_sam")
            precision_level=task.input_data.get("precision_level","high")
            image_path=task.input_data.get("image_path","")
            retry_count=task.input_data.get("retry_count",0)
            previous_issues=task.input_data.get("previous_issues",[])
            logger.info(f"分割方法: {segmentation_method},重试次数:{retry_count}")
            logger.info(f"目标对象: {target_subject}, 目标部位: {target_parts}")
            logger.info(f"精度要求: {precision_level}")
            if retry_count>0:
                segmentation_method,previous_level,instruction=self._adjust_parameters_for_retry\
                    (previous_issues, segmentation_method, precision_level, instruction)
                logger.info(f"重试调整后 - 方法: {segmentation_method}, 精度: {precision_level}")
            #构建文本提示
            text_prompt=self._build_text_prompt(target_subject,target_parts,instruction)
            logger.info(f"构建文本提示:{text_prompt}")
            segmentation_func=self.supported_methods.get(segmentation_method)
            result=await segmentation_func(image_path,text_prompt,precision_level)
            processed_result=self._post_process_segmentation(result,target_parts)
            # 评估分割质量
            quality_assessment = self._assess_segmentation_quality(processed_result)
            logger.info(f"分割质量评估: {quality_assessment}")

            # 将质量评估结果添加到处理结果中
            processed_result["quality_assessment"] = quality_assessment
            processed_result["retry_count"] = retry_count
            if self._should_retry(quality_assessment,retry_count):
                return self._create_retry_result(processed_result,quality_assessment,task)
            else:
                success=quality_assessment["overall_score"]>=self.min_quality_threshold
                return WorkerResult(success=True,data=processed_result,
                feedback_to_planner=self._generate_planner_feedback(quality_assessment, success,retry_count))
        except Exception as e:
            logger.info(f"图像分割失败{e}")
            return WorkerResult(success=False,data=[],feedback_to_planner={
                "action":"task_failed",
                "reason":f"执行异常:{str(e)}",
                "recommendations":["检查输入参数","验证图像路径","确认分割方法可用性"]
            })
    def _should_retry(self,quality_assessment:dict,retry_count:int)->bool:
        quality_below_threshold=quality_assessment["overall_score"]<self.min_quality_threshold
        not_max_retries=retry_count<self.max_retry_attempts
        has_actionable_issues=bool(quality_assessment.get("recommendations"))
        return quality_below_threshold and not_max_retries and has_actionable_issues
    def _create_retry_result(self,processed_result:dict,quality_assessment:dict,original_task:Task)->WorkerResult:
        retry_feedback={
            "action":"retry_required",
            "reason":f"分割质量不达标(得分: {quality_assessment['overall_score']:.2f}, 阈值: {self.min_quality_threshold})",
            "current_issues":quality_assessment["issues"],
            "recommendations":quality_assessment["recommendations"],
            "suggested_adjustments":self._get_parameter_adjustments(quality_assessment["issues"]),
            "retry_task_data":self._prepare_retry_task_data(original_task,quality_assessment)
        }
        return WorkerResult(
            success=False,
            data=processed_result,
            feedback_to_planner=retry_feedback
        )

    def _prepare_retry_task_data(self, original_task: Task, quality_assessment: dict) -> dict:
        """准备重试任务数据"""
        retry_data = original_task.input_data.copy()
        retry_data["retry_count"] = retry_data.get("retry_count", 0) + 1
        retry_data["previous_issues"] = quality_assessment["issues"]
        retry_data["previous_score"] = quality_assessment["overall_score"]

        return retry_data
    def _adjust_parameters_for_retry(self, previous_issues: list, current_method: str,
                                     current_precision: str, current_instruction: str) -> tuple:
        """根据之前的问题调整参数"""
        new_method = current_method
        new_precision = current_precision
        new_instruction = current_instruction

        # 根据具体问题调整参数
        if "Low confidence scores" in previous_issues:
            # 尝试切换分割方法
            if current_method == "lang_sam":
                new_method = "dino"
            else:
                new_method = "lang_sam"
            logger.info(f"由于置信度低，切换分割方法从 {current_method} 到 {new_method}")

        if "Some regions too small" in previous_issues:
            # 降低精度要求以获得更大的区域
            precision_map = {"high": "medium", "medium": "low", "low": "low"}
            new_precision = precision_map.get(current_precision, "medium")
            logger.info(f"由于区域过小，调整精度从 {current_precision} 到 {new_precision}")

        if "No target parts detected" in previous_issues:
            # 增强文本提示
            new_instruction = f"{current_instruction} 请更精确地定位目标区域"
            logger.info("由于未检测到目标，增强文本提示")

        return new_method, new_precision, new_instruction
    def _build_text_prompt(self,subject:str,target_parts:list,instruction:str)->str:
        if not target_parts:
            return f"segment {subject} in the image"
        parts_str=" and ".join(target_parts)
        if "挥舞" in instuction or "wave" in instruction.lower():
            prompt=f"segment the {parts_str} of {subject} that will be used for waving motion"
        elif "转动" in instruction or "rotate" in instruction.lower():
            prompt=f"segment the {parts_str} of {subject} that will rotate"
        elif "摆动" in instruction or "swing" in instruction.lower():
            prompt=f"segment the {parts_str} of {subject} that will swing"
        else:
            prompt =f"segment the {parts_str} of {subject}"
        prompt+=",focus on precise boundaries"
        return prompt

    async def _execute_lang_sam(self, image_path: str, text_prompt: str, precision_level: str) -> dict:
        """执行Lang-SAM分割"""
        logger.info("使用Lang-SAM进行文本指引分割")

        # 模拟Lang-SAM调用过程
        await asyncio.sleep(2)  # 模拟分割时间

        # TODO: 实际调用Lang-SAM
        """
        实际实现应该是：

        import torch
        from lang_sam import LangSAM

        # 初始化Lang-SAM模型
        model = LangSAM()

        # 加载图像
        image = Image.open(image_path)

        # 执行分割
        masks, boxes, phrases, logits = model.predict(image, text_prompt)

        # 处理结果...
        """

        # 模拟返回结果
        mock_result = {
            "method": "lang_sam",
            "masks": self._generate_mock_masks(),
            "bounding_boxes": [[50, 50, 150, 150], [200, 100, 300, 200]],
            "confidence_scores": [0.92, 0.87],
            "phrases": ["left hand", "right hand"],
            "logits": [4.2, 3.8]
        }

        return mock_result

    async def _execute_dino(self, image_path: str, text_prompt: str, precision_level: str) -> dict:
        """执行DINO分割"""
        logger.info("使用DINO进行物体分割")

        await asyncio.sleep(1.5)  # 模拟分割时间

        # TODO: 实际调用DINO
        """
        实际实现应该是：

        from transformers import AutoImageProcessor, AutoModel
        import torch

        # 加载DINO模型
        processor = AutoImageProcessor.from_pretrained('facebook/dino-vitb16')
        model = AutoModel.from_pretrained('facebook/dino-vitb16')

        # 处理图像
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")

        # 获取特征
        with torch.no_grad():
            outputs = model(**inputs)

        # 基于特征进行分割...
        """

        # 模拟返回结果
        mock_result = {
            "method": "dino",
            "masks": self._generate_mock_masks(),
            "bounding_boxes": [[60, 60, 160, 160], [210, 110, 310, 210]],
            "confidence_scores": [0.89, 0.84],
            "phrases": ["detected region 1", "detected region 2"],
            "features": "mock_features"
        }

        return mock_result

    def _generate_mock_masks(self) -> list:
        """生成模拟的分割掩码数据"""
        # 模拟两个分割区域的像素坐标
        mask1_pixels = []
        mask2_pixels = []

        # 模拟第一个区域 (左手区域: 50x50 到 150x150)
        for y in range(50, 151):
            for x in range(50, 151):
                if (x - 100) ** 2 + (y - 100) ** 2 <= 2500:  # 圆形区域
                    mask1_pixels.append([x, y])

        # 模拟第二个区域 (右手区域: 200x100 到 300x200)
        for y in range(100, 201):
            for x in range(200, 301):
                if (x - 250) ** 2 + (y - 150) ** 2 <= 2500:  # 圆形区域
                    mask2_pixels.append([x, y])

        return [
            {
                "region_id": 1,
                "pixel_coordinates": mask1_pixels,
                "pixel_count": len(mask1_pixels),
                "center": [100, 100],
                "area": len(mask1_pixels)
            },
            {
                "region_id": 2,
                "pixel_coordinates": mask2_pixels,
                "pixel_count": len(mask2_pixels),
                "center": [250, 150],
                "area": len(mask2_pixels)
            }
        ]

    def _post_process_segmentation(self, raw_result: dict, target_parts: list) -> dict:
        """后处理分割结果"""
        processed_result = {
            "segmentation_method": raw_result.get("method", "unknown"),
            "segmented_regions": [],
            "total_regions": len(raw_result.get("masks", [])),
            "success": True
        }

        masks = raw_result.get("masks", [])
        boxes = raw_result.get("bounding_boxes", [])
        confidences = raw_result.get("confidence_scores", [])
        phrases = raw_result.get("phrases", [])

        # 处理每个分割区域
        for i, mask in enumerate(masks):
            region_info = {
                "region_id": i + 1,
                "target_part": phrases[i] if i < len(phrases) else f"region_{i + 1}",
                "pixel_coordinates": mask["pixel_coordinates"],
                "pixel_count": mask["pixel_count"],
                "bounding_box": {
                    "x1": boxes[i][0] if i < len(boxes) else 0,
                    "y1": boxes[i][1] if i < len(boxes) else 0,
                    "x2": boxes[i][2] if i < len(boxes) else 0,
                    "y2": boxes[i][3] if i < len(boxes) else 0,
                },
                "center_point": mask["center"],
                "area": mask["area"],
                "confidence": confidences[i] if i < len(confidences) else 0.0,
                "is_target_part": any(part in phrases[i] if i < len(phrases) else "" for part in target_parts)
            }

            processed_result["segmented_regions"].append(region_info)

        # 添加可视化信息
        processed_result["visualization"] = {
            "mask_overlay_generated": True,
            "colored_regions": True,
            "boundary_highlighted": True,
            "save_path": f"segmentation_result_{hash(str(target_parts)) % 10000}.png"
        }

        # 添加质量评估
        processed_result["quality_assessment"] = self._assess_segmentation_quality(processed_result)

        return processed_result

    def _get_parameter_adjustments(self, issues: list) -> dict:
        """获取参数调整建议"""
        adjustments = {}

        if "Low confidence scores" in issues:
            adjustments["segmentation_method"] = "尝试不同的分割方法"
        if "Some regions too small" in issues:
            adjustments["precision_level"] = "降低精度要求"
        if "No target parts detected" in issues:
            adjustments["text_prompt"] = "优化文本提示描述"

        return adjustments

    def _generate_planner_feedback(self, quality_assessment: dict, success: bool, retry_count: int) -> dict:
        """生成给 Planner 的反馈"""
        if success:
            return {
                "action": "proceed_to_next",
                "message": f"图像分割质量达标 (得分: {quality_assessment['overall_score']:.2f})",
                "retry_count": retry_count,
                "final_quality_score": quality_assessment['overall_score']
            }
        else:
            if retry_count >= self.max_retry_attempts:
                return {
                    "action": "max_retries_reached",
                    "message": f"已达最大重试次数 ({self.max_retry_attempts})，质量仍不达标",
                    "final_score": quality_assessment['overall_score'],
                    "final_issues": quality_assessment['issues'],
                    "recommendations": ["考虑更换输入图像", "调整任务参数", "使用人工干预"]
                }
            else:
                return {
                    "action": "quality_below_threshold",
                    "message": "分割质量低于阈值，建议终止或调整策略",
                    "score": quality_assessment['overall_score'],
                    "threshold": self.min_quality_threshold,
                    "issues": quality_assessment['issues']
                }

    def _assess_segmentation_quality(self, result: dict) -> dict:
        """评估分割质量"""
        regions = result.get("segmented_regions", [])

        if not regions:
            return {
                "overall_score": 0.0,
                "issues": ["No regions detected"],
                "average_confidence": 0.0,
                "average_size_score": 0.0,
                "recommendations": ["检查图像质量和文本提示准确性", "尝试不同的分割方法"]
            }

        total_confidence = sum(region.get("confidence", 0) for region in regions)
        avg_confidence = total_confidence / len(regions)

        # 检查区域大小合理性
        size_scores = []
        for region in regions:
            area = region.get("area", 0)
            if 100 <= area <= 10000:  # 合理的区域大小范围
                size_scores.append(1.0)
            elif area < 100:
                size_scores.append(0.5)  # 区域太小
            else:
                size_scores.append(0.7)  # 区域较大但可接受

        avg_size_score = sum(size_scores) / len(size_scores) if size_scores else 0.5

        # 综合评分
        overall_score = (avg_confidence * 0.6 + avg_size_score * 0.4)

        issues = []
        if avg_confidence < 0.7:
            issues.append("Low confidence scores")
        if any(region.get("area", 0) < 100 for region in regions):
            issues.append("Some regions too small")
        if len(regions) == 0:
            issues.append("No target parts detected")

        return {
            "overall_score": overall_score,
            "average_confidence": avg_confidence,
            "average_size_score": avg_size_score,
            "issues": issues,
            "recommendations": self._generate_recommendations(overall_score, issues)
        }


    def _generate_recommendations(self, score: float, issues: list) -> list:
        """生成改进建议"""
        recommendations = []

        if score < 0.6:
            recommendations.append("Consider using different segmentation method")
        if "Low confidence scores" in issues:
            recommendations.append("Refine text prompt for better targeting")
        if "Some regions too small" in issues:
            recommendations.append("Adjust precision level or merge small regions")
        if "No target parts detected" in issues:
            recommendations.append("Check image quality and text prompt accuracy")

        return recommendations

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

class QwenLLMClient:
    print(f"in qwen")
    def __init__(self,api_key="sk-f2dce495ff4c41509745ab5cfb6beba4",base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        print(f"in qweninit")
        self.client=openai.OpenAI(api_key=api_key,base_url=base_url)
    async def call_llm(self,prompt:str,system_prompt:str="")->str:
        try:
            messages=[]
            if system_prompt:
                messages.append({"role":"system","content":system_prompt})
            messages.append({"role":"user","content":prompt})

            response=self.client.chat.completions.create(
                model="qwen1.5-72b-chat",
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败{e}")
            return f"Error{e}"


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

class AgentSystem:
    def __init__(self):
        print(f"before llm_client")
        self.llm_client=QwenLLMClient()
        print(f"in llm_client")
        self.planner=Planner(self.llm_client)
        self.max_iterations=3
    async def process_user_instruction(self,instruction:str)->str:
        logger.info(f"接收用户指令:{instruction}")
        iteration=0
        while iteration<self.max_iterations:
            iteration+=1
            logger.info(f"开始第{iteration}次迭代执行")
            try:
                task_plan=await self.planner.plan_tasks_with_reasoning(instruction)
                logger.info(f"任务规划完成，生成{len(task_plan.tasks)}个任务")
                self._log_thought_chain(task_plan.thought_chain)

                results=await self.planner.execute_task_plan(task_plan)
                #need_improvement=await self.planner.evaluate_and_improve(results)
                # 3. 简化版评估（暂时不考虑反思机制）
                success = self._evaluate_results(results, task_plan)

                if success:
                    logger.info("任务执行成功，效果满意")
                    return self._generate_completion_message(results, task_plan)
                else:
                    logger.info(f"第{iteration}次迭代结果不满意，准备重新执行")
                    if iteration < self.max_iterations:
                        # 为下次迭代准备优化后的指令
                        instruction = self._optimize_instruction_for_retry(instruction, results, task_plan)
                        continue
                    else:
                        logger.warning("达到最大迭代次数，返回当前结果")
                        return self._generate_completion_message(results, task_plan, is_final=True)
            except Exception as e:
                logger.error(f"第{iteration}次迭代执行出错: {str(e)}")
                if iteration >= self.max_iterations:
                    return f"任务执行失败，已达最大重试次数。错误信息: {str(e)}"
                continue

        return "任务执行完成"

    def _optimize_instruction_for_retry(self, original_instruction: str, results: Dict[str, Any],
                                        task_plan: TaskPlan) -> str:
        """
        为重试优化指令
        根据失败的任务和结果，对指令进行微调
        """
        try:
            # 分析失败的任务
            failed_tasks = [task for task in task_plan.tasks if task.status == TaskStatus.FAILED]

            if failed_tasks:
                # 根据失败任务类型调整指令
                failed_types = [task.worker_type for task in failed_tasks]

                if "image_segmentation" in failed_types:
                    # 如果图像分割失败，可能需要更明确的描述
                    return f"{original_instruction}（请更精确地分割目标区域）"
                elif "pointcloud_mapping" in failed_types:
                    # 如果点云映射失败，可能需要降低复杂度
                    return f"{original_instruction}（使用更简单的映射策略）"
                elif "rigid_motion" in failed_types:
                    # 如果运动控制失败，可能需要调整运动参数
                    return f"{original_instruction}（使用更平滑的运动效果）"
                elif "visualization" in failed_types:
                    # 如果可视化失败，可能需要简化渲染
                    return f"{original_instruction}（使用基础可视化模式）"

            # 如果没有明确的失败模式，返回原指令
            return original_instruction

        except Exception as e:
            logger.error(f"指令优化出错: {str(e)}")
            return original_instruction

    def _generate_completion_message(self, results: Dict[str, Any], task_plan: TaskPlan,
                                     is_final: bool = False) -> str:
        """
        生成完成消息
        """
        try:
            message_parts = []

            if is_final:
                message_parts.append("任务执行已完成（已达最大迭代次数）")
            else:
                message_parts.append("任务执行成功完成")

            # 添加执行摘要
            completed_tasks = [task for task in task_plan.tasks if task.status == TaskStatus.COMPLETED]
            failed_tasks = [task for task in task_plan.tasks if task.status == TaskStatus.FAILED]

            message_parts.append(f"成功执行了{len(completed_tasks)}个任务")
            if failed_tasks:
                message_parts.append(f"有{len(failed_tasks)}个任务执行失败")

            # 添加主要结果信息
            if "task_visualization" in results:
                viz_result = results["task_visualization"]
                if isinstance(viz_result, dict):
                    status = viz_result.get("status", "unknown")
                    message_parts.append(f"可视化结果: {status}")

            # 添加思维链置信度摘要
            if task_plan.thought_chain:
                avg_confidence = sum(chain.confidence for chain in task_plan.thought_chain) / len(
                    task_plan.thought_chain)
                message_parts.append(f"平均置信度: {avg_confidence:.2f}")

            return "\n".join(message_parts)

        except Exception as e:
            logger.error(f"生成完成消息出错: {str(e)}")
            return "任务执行完成，但生成摘要时出现错误"



def show_help_examples():
    """显示帮助和示例"""
    print("\n" + "=" * 60)
    print("📚 指令示例和帮助")
    print("=" * 60)

    examples = [
        {
            "category": "🐿️ 动物动作控制",
            "examples": [
                "让松鼠挥舞手臂",
                "让猫摆动尾巴",
                "让狗摇头",
                "让人点头"
            ]
        },
        {
            "category": "🎭 复杂动作组合",
            "examples": [
                "让松鼠先点头然后挥舞手臂",
                "让猫一边摆尾巴一边转头",
                "让人做跳跃动作"
            ]
        },
        {
            "category": "🎨 特定部位控制",
            "examples": [
                "只让松鼠的头部转动",
                "让猫的前爪做挥舞动作",
                "控制人的手臂做摆动"
            ]
        }
    ]

    for category_info in examples:
        print(f"\n{category_info['category']}:")
        for i, example in enumerate(category_info['examples'], 1):
            print(f"  {i}. {example}")

    print("\n" + "=" * 60)
    print("💡 提示:")
    print("• 使用自然语言描述即可，系统会智能解析")
    print("• 支持中文指令，描述越详细效果越好")
    print("• 系统会自动选择最适合的处理策略")
    print("=" * 60)
async def main():
    """主函数 - 多模态点云控制Agent系统"""
    print("=" * 70)
    print("🎯 多模态点云控制Agent系统 v2.0")
    print("=" * 70)
    print("系统特性:")
    print("• 智能思维链分析")
    print("• 自适应任务规划")
    print("• 多模态处理流程")
    print("• 实时可视化反馈")
    print("=" * 70)

    # 初始化系统
    try:
        print("\n🔧 正在初始化系统...")
        # 注意：请替换为实际的API密钥和地址
        system = AgentSystem()
        print("✅ 系统初始化完成")

        # 显示使用提示
        print("\n📖 使用指南:")
        print("• 支持自然语言描述动作指令")
        print("• 例如: '让松鼠挥舞手臂'、'让猫摆动尾巴'")
        print("• 输入 'help' 查看更多示例")
        print("• 输入 'quit' 或 'exit' 退出系统")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 系统初始化失败: {e}")
        logger.error(f"系统初始化失败: {e}")
        return

    session_count = 0

    while True:
        try:
            # 获取用户输入
            instruction = input(f"\n[会话 {session_count + 1}] 请输入控制指令: ").strip()

            # 处理特殊命令
            if instruction.lower() in ['quit', 'exit', 'q']:
                print("\n👋 感谢使用多模态点云控制系统，再见！")
                break

            if instruction.lower() == 'help':
                show_help_examples()
                continue

            if not instruction:
                print("⚠️  请输入有效的指令")
                continue

            # 处理用户指令
            session_count += 1
            print(f"\n🚀 正在处理您的指令: '{instruction}'")
            print("📊 执行流程: 思维链分析 → 任务规划 → 执行处理 → 结果评估")
            print("-" * 60)

            start_time = asyncio.get_event_loop().time()

            # 调用更新后的处理方法
            completion_message = await system.process_user_instruction(instruction)

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # 显示结果
            print("\n" + "=" * 60)
            print("📋 任务完成报告")
            print("=" * 60)
            print(completion_message)
            print("-" * 60)
            print(f"⏱️  执行耗时: {execution_time:.2f}秒")
            print(f"🔢 会话编号: {session_count}")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n⚠️  系统被用户中断")
            user_choice = input("是否要退出系统? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes']:
                print("👋 系统退出")
                break
            else:
                print("继续运行...")
                continue

        except Exception as e:
            session_count += 1  # 即使出错也计入会话
            print(f"\n❌ 系统执行错误: {e}")
            print("💡 建议:")
            print("  • 检查指令格式是否正确")
            print("  • 确认网络连接正常")
            print("  • 尝试重新输入指令")
            logger.error(f"会话{session_count}执行错误: {e}")

            # 询问是否继续
            continue_choice = input("\n是否继续使用系统? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '']:
                print("👋 系统退出")
                break


if __name__ == "__main__":
    asyncio.run(main())