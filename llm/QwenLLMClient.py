import openai
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