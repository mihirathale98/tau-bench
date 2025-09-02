from litellm import completion
from openai import OpenAI
import os

openai_client = OpenAI()


def call_completion(**kwargs):
    
    if kwargs.get("custom_llm_provider") == "hosted_vllm":
        kwargs.pop("custom_llm_provider")
        openai_client.base_url = kwargs.pop("base_url")
        openai_client.api_key = kwargs.pop("api_key")
        response = openai_client.chat.completions.create(**kwargs)
        return response
    else:
        kwargs.pop("base_url")
        kwargs.pop("api_key")
        return completion(**kwargs)
