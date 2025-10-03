import openai
from litellm import completion

# client = openai.OpenAI()


# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "user", "content": "What is the capital of France?"}
#     ],
#     n=5
# )

vertex_ai_kwargs = {
                    "model": f"vertex_ai/claude-sonnet-4@20250514",
                    "temperature": 1,
                    "vertex_ai_project": "itpc-gcp-ai-eng-claude",
                    "vertex_ai_location": "us-east5",
                    # "vertex_credentials": json.dumps(json.load(open("/path/to/your/service-account-key.json"))
                }

response = completion(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    n=1,
    **vertex_ai_kwargs,
)

# response = completion(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "user", "content": "What is the capital of France?"}
#     ],
#     n=5,
# )
print(response)
print(response.usage.model_dump())