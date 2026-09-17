import os

from openai import OpenAI

base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing API key: set LLM_API_KEY or OPENAI_API_KEY.")
model = os.getenv("LLM_MODEL", "qwen-plus")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

stream = True # or False
max_tokens = 3000

response_format = { "type": "text" }

chat_completion_res = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Output 'Hello'",
        }
    ],
    stream=stream,
    max_tokens=max_tokens,
    stream_options={"include_usage": True} if stream else None,
    extra_body={"enable_thinking": False}
)

if stream:
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    for chunk in chat_completion_res:
        # 打印流式文本
        if getattr(chunk, "choices", None):
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="")

        # 某些兼容接口会在最后一个 chunk 返回 usage
        if getattr(chunk, "usage", None) is not None:
            prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
            completion_tokens = getattr(chunk.usage, "completion_tokens", None)
            total_tokens = getattr(chunk.usage, "total_tokens", None)

    print()

    if total_tokens is not None:
        print(f"prompt_tokens: {prompt_tokens}")
        print(f"completion_tokens: {completion_tokens}")
        print(f"total_tokens: {total_tokens}")
    else:
        print("该流式接口未返回 usage，无法直接统计 token 用量。")
        print("可改用 stream=False 再请求一次，以获取 usage。")

else:
    print(chat_completion_res.choices[0].message.content)

    if getattr(chat_completion_res, "usage", None) is not None:
        print(f"prompt_tokens: {chat_completion_res.usage.prompt_tokens}")
        print(f"completion_tokens: {chat_completion_res.usage.completion_tokens}")
        print(f"total_tokens: {chat_completion_res.usage.total_tokens}")
    else:
        print("该接口未返回 usage，无法直接统计 token 用量。")
