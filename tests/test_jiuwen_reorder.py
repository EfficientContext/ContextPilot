"""
With ContextPilot: docs reordered for maximum KV-cache prefix reuse.
Restart llama-server to clear the cache before running this, then compare
the time against test_jiuwen_baseline.py.
"""
import os
import time
import asyncio
import contextpilot as cp

from openjiuwen.core.workflow import Start, End, LLMComponent, LLMCompConfig, generate_workflow_key
from openjiuwen.core.foundation.llm import ModelRequestConfig, ModelClientConfig
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.single_agent.legacy import WorkflowAgentConfig
from openjiuwen.core.application.workflow_agent import WorkflowAgent
from openjiuwen.core.workflow import Workflow, WorkflowCard

os.environ.setdefault("API_BASE", "http://localhost:8889/v1")
os.environ.setdefault("API_KEY", "EMPTY")
os.environ.setdefault("MODEL_PROVIDER", "openai")
os.environ.setdefault("MODEL_NAME", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
os.environ.setdefault("LLM_SSL_VERIFY", "false")

model_client_config = ModelClientConfig(
    client_provider=os.getenv("MODEL_PROVIDER"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE"),
    verify_ssl=os.getenv("LLM_SSL_VERIFY").lower() == "true",
    timeout=120.0,
    max_retries=1,
)
model_config = ModelRequestConfig(model=os.getenv("MODEL_NAME"), max_tokens=200)

workflow_card = WorkflowCard(
    id="generate_text_workflow",
    name="generate_text",
    version="1.0",
    description="根据用户输入生成文本",
    input_params={
        "type": "object",
        "properties": {
            "query":   {"type": "string", "description": "用户输入"},
            "context": {"type": "string", "description": "检索到的上下文"},
        },
        "required": ["query"]
    }
)

flow = Workflow(card=workflow_card)
start = Start()
end = End({"responseTemplate": "工作流输出文本: {{output}}"})

llm_config = LLMCompConfig(
    model_client_config=model_client_config,
    model_config=model_config,
    template_content=[
        {
            "role": "system",
            "content": "你是一个AI助手，能够帮我完成任务。\n\n相关上下文：\n{{context}}"
        },
        {"role": "user", "content": "{{query}}"}
    ],
    response_format={"type": "text"},
    output_config={
        "output": {"type": "string", "description": "大模型输出"}
    }
)
llm = LLMComponent(llm_config)

flow.set_start_comp("start", start, inputs_schema={"query": "${query}", "context": "${context}"})
flow.add_workflow_comp("llm", llm, inputs_schema={"query": "${start.query}", "context": "${start.context}"})
flow.set_end_comp("end", end, inputs_schema={"output": "${llm.output}"})
flow.add_connection("start", "llm")
flow.add_connection("llm", "end")

Runner.resource_mgr.add_workflow(
    WorkflowCard(id=generate_workflow_key(flow.card.id, flow.card.version)),
    lambda: flow,
)

agent_config = WorkflowAgentConfig(
    id="hello_agent",
    version="0.1.1",
    description="第一个Agent",
)
workflow_agent = WorkflowAgent(agent_config)
workflow_agent.add_workflows([flow])

cp_instance = cp.ContextPilot(use_gpu=False)


def get_reordered_context(query: str, retrieved_docs: list[str]) -> str:
    messages = cp_instance.optimize(retrieved_docs, query)
    for msg in messages:
        if msg["role"] == "system":
            return msg["content"]
    return "\n\n".join(retrieved_docs)


async def main():
    query = "你好，请生成一则笑话，不要超过20个字"
    retrieved_docs = [
        "笑话通常由铺垫和出人意料的结尾组成。",
        "幽默是人类沟通的重要方式。",
        "简短的笑话更容易让人记住。",
    ]

    context = get_reordered_context(query, retrieved_docs)
    print(f"[ContextPilot] Reordered Context: {context}")

    t0 = time.perf_counter()
    invoke_result = await Runner.run_agent(
        workflow_agent,
        {"query": query, "context": context}
    )
    elapsed = time.perf_counter() - t0

    output_result = invoke_result.get("output").result
    output = output_result.get("response") or str(output_result)
    print(f"[ContextPilot] Time: {elapsed:.3f}s")
    print(f"[ContextPilot] Output: {output}")


asyncio.run(main())
