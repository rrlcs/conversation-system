import together
from langchain.llms.base import LLM
from pydantic import Field
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# load_dotenv()
# TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

class DeepSeekLLM(LLM):
    model: str = "deepseek-ai/DeepSeek-V3"
    # Define callbacks as a normal field with a default factory.
    callbacks: list = Field(default_factory=list, exclude=True)
    
    def __init__(self, api_key: str):
        together.api_key = api_key
        super().__init__()

    def _call(self, prompt: str, stop=None) -> str:
        
        print("\n========== [DEBUG] Prompt Sent to LLM ==========")
        print(prompt)  # 🚀 See the actual input
        print("=================================================\n")
        # Call the Together API with the prompt and our DeepSeek model.
        response = together.Complete.create(
            prompt=prompt,
            model=self.model,
            max_tokens=200
        )
        # Extract and return the generated text.
        return response["choices"][0]["text"].strip()

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "deepseek_chat"

# Rebuild the model so that pydantic is aware of the new fields.
DeepSeekLLM.model_rebuild()
