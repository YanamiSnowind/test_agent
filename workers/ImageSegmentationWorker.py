#from workers import BaseWorker
from TypeEnums import BaseWorker
from TypeEnums import *
import asyncio

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
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