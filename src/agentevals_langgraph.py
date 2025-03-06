import json

from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import (
    extract_langgraph_trajectory_from_thread,
)
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

load_dotenv(override=True)


@tool
def search(query: str):
    """Call to surf the web."""
    user_answer = interrupt("Tell me the answer to the question.")
    return user_answer


tools = [search]

checkpointer = MemorySaver()
graph = create_react_agent(
    model="gpt-4o-mini",
    checkpointer=checkpointer,
    tools=[search],
)

graph.invoke(
    {"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    config={"configurable": {"thread_id": "1"}},
)
# Resume the agent with a new command, simulating a human-in-the-loop workflow
graph.invoke(
    Command(resume="It is rainy and 70 degrees!"),
    config={"configurable": {"thread_id": "1"}},
)

# Extract the trajectory from the first two thread runs
extracted_trajectory = extract_langgraph_trajectory_from_thread(
    graph,
    {"configurable": {"thread_id": "1"}},
)

print(json.dumps(extracted_trajectory, indent=2))

graph_trajectory_evaluator = create_graph_trajectory_llm_as_judge(
    model="openai:o3-mini",
)

res = graph_trajectory_evaluator(
    inputs=extracted_trajectory["inputs"],
    outputs=extracted_trajectory["outputs"],
)

print(res)
