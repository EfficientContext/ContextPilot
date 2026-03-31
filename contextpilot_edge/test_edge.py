from openai import OpenAI

# llama-server exposes a native /v1/* OpenAI-compatible API — point at it directly.
# If you want cache-reuse metrics + GPU monitoring, start proxy_server.py on
# :8890 and change the base_url to http://localhost:8890/v1 instead.
client = OpenAI(
    base_url="http://localhost:8889/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="llama.cpp",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)