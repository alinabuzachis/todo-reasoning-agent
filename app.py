from typing import Any, List
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import os
import json
from pydantic import BaseModel

load_dotenv(override=True)


# Tool Definitions

create_todos_json = {
    "name": "create_todos",
    "description": "Create a new todo",
    "parameters": {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "The description of the todo",
                },
            }
        },
        "required": ["descriptions"],
        "additionalProperties": False,
    },
}

mark_complete_json = {
    "name": "mark_complete",
    "description": "Mark a todo as complete",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "The index of the todo to mark as complete",
            },
            "completion_notes": {
                "type": "string",
                "description": "The notes about the completion of the todo",
            },
        },
        "required": ["index"],
        "additionalProperties": False,
    },
}

# Prompts

system_message = """
You are given a problem to solve, by using your todo tools to plan a list of steps, then carrying out each step in turn.
Now use the todo list tools, create a plan, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Rich console markup without code blocks.
If you have the answer, your response must be plain text, not a tool call.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""

evaluator_system_prompt = """
You are an evaluator that decides whether a response to a todo list is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's
latest response is corrrect and acceptable. Provide your evaluation in Rich console markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


class ToDoHandler:
    def __init__(self):
        self.client = OpenAI()
        self.worker_model = os.getenv("WORKER_MODEL", "llama3.2:1b")
        self.reasoner_model = os.getenv("REASONER_MODEL", "deepseek-r1:1.5b")
        self.to_do = []
        self.completed = []
        self.tools = [
            {"type": "function", "function": create_todos_json},
            {"type": "function", "function": mark_complete_json},
        ]

    def create_todos(self, descriptions):
        self.to_do = [{"task": d, "done": False} for d in descriptions]
        return {"status": "success", "tasks": self.to_do}

    def mark_complete(self, index, completion_notes=""):
        try:
            self.to_do[index]["done"] = True
            return {"status": "updated", "notes": completion_notes}
        except (IndexError, TypeError):
            return {"error": "Invalid index"}

    def handle_tool_calls(self, tool_calls) -> List[dict[str, Any]]:
        results: List[dict] = []
        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            # Map string names to class methods
            func = getattr(self, tool_call.function.name)
            content = func(**args)
            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(content),
                    "tool_call_id": tool_call.id,
                }
            )
        return results

    def loop(self, messages: List[dict[str, Any]]) -> str:
        done: bool = False
        while not done:
            response = self.client.chat.completions.create(
                model=self.worker_model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="none",
            )
            message = response.choices[0].message
            finish_reason: str = response.choices[0].finish_reason
            if finish_reason == "tool_calls":
                # 1. Add the Assistant's tool call to history
                messages.append(message)
                # 2. Execute tools and get results
                results = self.handle_tool_calls(message.tool_calls)
                # 3. Add those tool results (role: tool) to history
                messages.extend(results)
            else:
                done = True
        return message.content

    @staticmethod
    def evaluator_user_prompt(
        reply: str, message: str, history: List[dict[str, Any]]
    ) -> str:
        user_prompt = (
            f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        )
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt

    def evaluate(
        self, reply: str, message: str, history: List[dict[str, Any]]
    ) -> Evaluation:
        messages = [
            {"role": "system", "content": evaluator_system_prompt},
            {
                "role": "user",
                "content": self.evaluator_user_prompt(reply, message, history),
            },
        ]

        response = self.client.chat.completions.parse(
            model=self.reasoner_model, messages=messages, response_format=Evaluation
        )
        return response.choices[0].message.parsed

    def rerun(
        self, reply: str, message: str, history: List[dict[str, Any]], feedback: str
    ) -> str:
        updated_system_prompt = (
            evaluator_system_prompt
            + f"\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        )
        updated_system_prompt += f"## Your attempted answer: {reply}\n\n"
        updated_system_prompt += f"## Reason for rejection: {feedback}\n\n"
        messages = [
            {"role": "system", "content": updated_system_prompt},
            {
                "role": "user",
                "content": self.evaluator_user_prompt(reply, message, history),
            },
        ]
        return self.loop(messages)

    def chat(self, user_message: str, history: list, max_retries: int = 2) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        for attempt in range(max_retries + 1):
            reply = self.loop(messages)
            evaluation = self.evaluate(reply, user_message, messages)
            if evaluation.is_acceptable:
                return reply

            # If not acceptable and we have retries left, update messages and loop again
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous answer was rejected. Feedback: {evaluation.feedback}. Please try again.",
                    }
                )
            else:
                # If we've exhausted retries, return the last reply with a disclaimer
                # or just the feedback.
                return f"Note: Could not reach an optimal solution after {max_retries} retries.\n\n{reply}"


if __name__ == "__main__":
    todo = ToDoHandler()
    gr.ChatInterface(todo.chat).launch()
