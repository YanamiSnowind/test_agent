"""
基于Agent控制的系统
"""
import json
import asyncio
import logging
from typing import Dict,List,Any,Optional
from llm import QwenLLMClient
from Planner import Planner
from dataclasses import dataclass
from enum import Enum
from abc import ABC,abstractmethod

import openai

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)


class AgentSystem:
    def __init__(self,llm_api_key:str="",llm_base_url:str=""):
        self.llm_client=QwenLLMClient(llm_api_key,llm_base_url)
        self.planner=Planner(self.llm_client)
        self.max_iterations=3
    #拆分任务并制定规划步骤
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