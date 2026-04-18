from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=512
    )
    return HuggingFacePipeline(pipeline=pipe)