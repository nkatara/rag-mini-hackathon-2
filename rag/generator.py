from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def load_llm():
    pipe = pipeline(
        task="text-generation",  
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)