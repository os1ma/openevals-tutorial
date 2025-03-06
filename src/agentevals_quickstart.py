import json

from agentevals.trajectory.llm import (
    TRAJECTORY_ACCURACY_PROMPT,
    create_trajectory_llm_as_judge,
)
from dotenv import load_dotenv

load_dotenv(override=True)

trajectory_evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    model="openai:o3-mini",
)

# This is a fake trajectory, in reality you would run your agent to get a real trajectory
outputs = [
    {"role": "user", "content": "What is the weather in SF?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "SF"}),
                },
            },
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
]

eval_result = trajectory_evaluator(
    outputs=outputs,
)

print(eval_result)
