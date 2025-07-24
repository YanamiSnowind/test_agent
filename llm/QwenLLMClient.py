import openai
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
import openai
print("openai:", openai)
print("openai path:", openai.__file__)
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