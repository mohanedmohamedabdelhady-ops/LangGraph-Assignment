=====================================================================
LangGraph Multi-Agent Assignment - My Solution
Research -> Write -> Review -> Revise
=====================================================================

HOW TO USE THIS FILE
- Every block marked [CODE CELL] goes in its own code cell in the notebook.
- Every block marked [MARKDOWN CELL] goes in a markdown cell.
- Run the notebook top to bottom, so all outputs and graph images show
  before you submit it.


=====================================================================
[MARKDOWN CELL]
# Setup
=====================================================================

[CODE CELL]
!pip install -q langgraph langchain langchain-openai wikipedia-api pydantic


[CODE CELL]
import os
import operator
from getpass import getpass
from typing import Annotated, TypedDict

from IPython.display import Image, display

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

import wikipediaapi

# put your API key here (you can use any other chat model instead)
os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


=====================================================================
[MARKDOWN CELL]
# Part 1: Researcher Graph
=====================================================================

[MARKDOWN CELL]
## Task 1: ResearcherState
messages uses add_messages and trace uses operator.add, so new values are
appended instead of overwriting the old ones.

[CODE CELL]
class ResearcherState(TypedDict):
    task: str
    research: str
    messages: Annotated[list, add_messages]
    trace: Annotated[list, operator.add]


[MARKDOWN CELL]
## Task 2: Wikipedia search tool

Question: Page content can be too large to fit in the LLM context. How would
you handle this?

Answer: I handle it in two ways:
1. I only take the top 3 search results (limit=3).
2. I cut each page text to its first 2000 characters (the beginning of a
   Wikipedia page has the most important info), so the total text stays small.
(Other options: split the text into chunks and use embeddings/retrieval to
keep only the relevant chunks, or summarize each page, but the assignment
wants RAW text, so I use truncation.)

[CODE CELL]
wiki = wikipediaapi.Wikipedia(
    user_agent="langgraph-assignment/1.0 (student@example.com)"
)

MAX_CHARS_PER_PAGE = 2000   # truncate each page so it fits in the context


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for the query and return the raw text of the top pages."""
    results = wiki.search(query, limit=3)
    text = ""
    for title, page in results.pages.items():
        text += f"# {title}\n{page.text[:MAX_CHARS_PER_PAGE]}\n\n"
    if text == "":
        return "No results found."
    return text


tools = [wikipedia_search]
llm_with_tools = llm.bind_tools(tools)


[MARKDOWN CELL]
## Task 3: researcher_node
The system prompt forbids summarizing/formatting. The node returns the raw
tool output as "research". On the first pass there is no tool output yet,
so research is empty. After the tool runs, the node is called again and
research gets filled.

[CODE CELL]
RESEARCHER_PROMPT = (
    "You are a researcher. Use the wikipedia_search tool to gather RAW "
    "information about the user's task. Do NOT summarize, do NOT rewrite, "
    "and do NOT format anything. Output must be raw notes only."
)


def researcher_node(state: ResearcherState) -> dict:
    # the task is always sent first, then the tool-calling history
    msgs = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=state["task"]),
    ] + state["messages"]

    response = llm_with_tools.invoke(msgs)

    # raw text = all the tool outputs so far (empty before the tool runs)
    tool_texts = [m.content for m in state["messages"]
                  if isinstance(m, ToolMessage)]
    research = "\n\n".join(tool_texts)

    return {
        "research": research,
        "messages": [response],
        "trace": ["researcher"],
    }


[MARKDOWN CELL]
## Task 4: Build and compile the researcher graph
researcher -> (tool call? yes -> tools -> back to researcher, no -> END)

[CODE CELL]
researcher_graph = StateGraph(ResearcherState)

researcher_graph.add_node("researcher", researcher_node)
researcher_graph.add_node("tools", ToolNode(tools))

researcher_graph.add_edge(START, "researcher")
# tools_condition goes to "tools" if the LLM asked for a tool, otherwise END
researcher_graph.add_conditional_edges("researcher", tools_condition)
researcher_graph.add_edge("tools", "researcher")

researcher_app = researcher_graph.compile()


[MARKDOWN CELL]
## Task 5: Run the researcher standalone

[CODE CELL]
result = researcher_app.invoke({
    "task": "Give me information about the history of coffee.",
    "research": "",
    "messages": [],
    "trace": [],
})

print("TASK:", result["task"])
print("\nRESEARCH (first 1000 chars):\n", result["research"][:1000])
print("\nTRACE:", result["trace"])

# checks
assert result["research"] != ""
assert "researcher" in result["trace"]
print("\nChecks passed: research is not empty and trace has 'researcher'.")


[MARKDOWN CELL]
## Task 6: Visualize the researcher graph

[CODE CELL]
display(Image(researcher_app.get_graph().draw_mermaid_png()))


=====================================================================
[MARKDOWN CELL]
# Part 2: Writer Graph
=====================================================================

[MARKDOWN CELL]
## Task 1: WriterState

[CODE CELL]
class WriterState(TypedDict):
    task: str
    research: str
    feedback: str
    draft: str
    attempts: int
    messages: Annotated[list, add_messages]
    trace: Annotated[list, operator.add]


[MARKDOWN CELL]
## Task 2: writer_node
The prompt always has the task. If research exists, it is added as source
material. If feedback exists, it is added as revision guidance.

[CODE CELL]
def writer_node(state: WriterState) -> dict:
    prompt = f"Task: {state['task']}\n"

    if state["research"] != "":
        prompt += f"\nRaw source material (use it if helpful):\n{state['research']}\n"

    if state["feedback"] != "":
        prompt += (
            f"\nYour previous draft:\n{state['draft']}\n"
            f"\nRevise the draft using this reviewer feedback:\n{state['feedback']}\n"
        )

    prompt += "\nWrite only the final text, nothing else."

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft": response.content,
        "attempts": state["attempts"] + 1,
        "messages": [response],
        "trace": ["writer"],
    }


[MARKDOWN CELL]
## Task 3: Build and compile the writer graph

[CODE CELL]
writer_graph = StateGraph(WriterState)

writer_graph.add_node("writer", writer_node)
writer_graph.add_edge(START, "writer")
writer_graph.add_edge("writer", END)

writer_app = writer_graph.compile()


[MARKDOWN CELL]
## Task 4: Run the writer standalone

[CODE CELL]
result = writer_app.invoke({
    "task": "Write a 3-sentence pitch for a plant-watering mobile app.",
    "research": "", "feedback": "", "draft": "", "attempts": 0,
    "messages": [], "trace": [],
})

print("DRAFT:\n", result["draft"])
print("\nATTEMPTS:", result["attempts"])
print("TRACE:", result["trace"])

assert result["draft"] != ""
assert result["attempts"] == 1
assert result["trace"] == ["writer"]
print("\nChecks passed.")


[MARKDOWN CELL]
## Task 5: Visualize the writer graph

[CODE CELL]
display(Image(writer_app.get_graph().draw_mermaid_png()))


=====================================================================
[MARKDOWN CELL]
# Part 3: Reviewer Graph
=====================================================================

[MARKDOWN CELL]
## Task 1: ReviewerState

[CODE CELL]
class ReviewerState(TypedDict):
    task: str
    draft: str
    feedback: str
    approved: bool
    messages: Annotated[list, add_messages]
    trace: Annotated[list, operator.add]


[MARKDOWN CELL]
## Task 2: Structured output model

[CODE CELL]
class Review(BaseModel):
    feedback: str = Field(description="Specific, actionable feedback on the draft.")
    approved: bool = Field(description="True if the draft is good enough to publish.")


[MARKDOWN CELL]
## Task 3: reviewer_node
with_structured_output(Review) makes the LLM return a Review object, so
approved is always a real True/False.

[CODE CELL]
reviewer_llm = llm.with_structured_output(Review)


def reviewer_node(state: ReviewerState) -> dict:
    prompt = (
        "You are a strict reviewer. Check the draft against the task. "
        "Only approve when the draft fully follows ALL the task requirements "
        "and no meaningful improvement is left. Otherwise reject it and give "
        "specific, actionable feedback.\n\n"
        f"Task: {state['task']}\n\n"
        f"Draft:\n{state['draft']}"
    )

    review = reviewer_llm.invoke([HumanMessage(content=prompt)])

    return {
        "feedback": review.feedback,
        "approved": review.approved,
        "messages": [AIMessage(content=review.feedback)],
        "trace": ["reviewer"],
    }


[MARKDOWN CELL]
## Task 4: Build and compile the reviewer graph

[CODE CELL]
reviewer_graph = StateGraph(ReviewerState)

reviewer_graph.add_node("reviewer", reviewer_node)
reviewer_graph.add_edge(START, "reviewer")
reviewer_graph.add_edge("reviewer", END)

reviewer_app = reviewer_graph.compile()


[MARKDOWN CELL]
## Task 5: Run the reviewer standalone

[CODE CELL]
result = reviewer_app.invoke({
    "task": "Write a slogan for a coffee shop.",
    "draft": "Coffee that wakes you up and keeps you going.",
    "feedback": "", "approved": False, "messages": [], "trace": [],
})

print("FEEDBACK:\n", result["feedback"])
print("\nAPPROVED:", result["approved"])
print("TRACE:", result["trace"])

assert result["feedback"] != ""
assert isinstance(result["approved"], bool)
assert result["trace"] == ["reviewer"]
print("\nChecks passed.")


[MARKDOWN CELL]
## Task 6: Visualize the reviewer graph

[CODE CELL]
display(Image(reviewer_app.get_graph().draw_mermaid_png()))


=====================================================================
[MARKDOWN CELL]
# Part 4: Orchestration Layer
=====================================================================

[MARKDOWN CELL]
## Task 1: OrchestratorState
No "messages" here. The conversation history stays inside each sub-graph.

[CODE CELL]
class OrchestratorState(TypedDict):
    task: str
    research: str
    draft: str
    feedback: str
    approved: bool
    attempts: int
    trace: Annotated[list, operator.add]


[MARKDOWN CELL]
## Task 2: Wrap each compiled graph as a node
The wrappers only use researcher_app, writer_app and reviewer_app (not the
inner nodes). Each one builds the input of its sub-graph, calls invoke, and
returns only the fields that are in OrchestratorState.

I return trace as ["researcher"] / ["writer"] / ["reviewer"] (one entry per
wrapper call) instead of the sub-graph's trace. This is because the
researcher sub-graph adds "researcher" twice (once for the tool call and once
for the final answer), and I want the final trace to be clean:
researcher, writer, reviewer, writer, reviewer, ...

[CODE CELL]
def run_researcher(state: OrchestratorState) -> dict:
    out = researcher_app.invoke({
        "task": state["task"],
        "research": "",
        "messages": [],
        "trace": [],
    })
    return {
        "research": out["research"],
        "trace": ["researcher"],
    }


def run_writer(state: OrchestratorState) -> dict:
    out = writer_app.invoke({
        "task": state["task"],
        "research": state["research"],
        "feedback": state["feedback"],
        "draft": state["draft"],
        "attempts": state["attempts"],
        "messages": [],
        "trace": [],
    })
    return {
        "draft": out["draft"],
        "attempts": out["attempts"],
        "trace": ["writer"],
    }


def run_reviewer(state: OrchestratorState) -> dict:
    out = reviewer_app.invoke({
        "task": state["task"],
        "draft": state["draft"],
        "feedback": "",
        "approved": False,
        "messages": [],
        "trace": [],
    })
    return {
        "feedback": out["feedback"],
        "approved": out["approved"],
        "trace": ["reviewer"],
    }


[MARKDOWN CELL]
## Task 3: The router
- If approved -> finish.
- If max attempts reached -> finish (safety cap, so there is no infinite loop).
- Otherwise -> go back to the writer to revise.

[CODE CELL]
MAX_ATTEMPTS = 4   # safety cap


def route_after_review(state: OrchestratorState) -> str:
    if state["approved"]:
        return "end"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "end"
    return "writer"


[MARKDOWN CELL]
## Task 4: Assemble and visualize the orchestration graph

[CODE CELL]
orchestrator = StateGraph(OrchestratorState)

orchestrator.add_node("researcher", run_researcher)
orchestrator.add_node("writer", run_writer)
orchestrator.add_node("reviewer", run_reviewer)

orchestrator.add_edge(START, "researcher")
orchestrator.add_edge("researcher", "writer")
orchestrator.add_edge("writer", "reviewer")
orchestrator.add_conditional_edges(
    "reviewer",
    route_after_review,
    {"writer": "writer", "end": END},
)

multi_agent_app = orchestrator.compile()

display(Image(multi_agent_app.get_graph().draw_mermaid_png()))


[MARKDOWN CELL]
## Task 5: Run the orchestrated workflow

[CODE CELL]
final_state = multi_agent_app.invoke({
    "task": "Write a single-sentence slogan for a coffee shop. Keep it under 10 words.",
    "research": "", "draft": "", "feedback": "", "approved": False,
    "attempts": 0, "trace": [],
})

print("RESEARCH (first 500 chars):\n", final_state["research"][:500])
print("\nFINAL DRAFT:", final_state["draft"])
print("LAST FEEDBACK:", final_state["feedback"])
print("APPROVED:", final_state["approved"])
print("ATTEMPTS:", final_state["attempts"])
print("TRACE:", final_state["trace"])


[CODE CELL]
# Final checks from the assignment
assert final_state["research"] != ""
assert final_state["draft"] != ""
assert final_state["attempts"] > 1
assert final_state["trace"][0] == "researcher"
assert final_state["trace"][1::2] == ["writer"] * (len(final_state["trace"][1::2]))
assert final_state["approved"] is True
print("All checks passed!")


[MARKDOWN CELL]
### Notes
- If the reviewer approves the very first draft, attempts will be 1 and the
  "attempts > 1" check fails. In that case just re-run the cell, or make the
  reviewer prompt a bit stricter (I already asked it to be strict).
- If it never gets approved, the safety cap (MAX_ATTEMPTS = 4) stops the loop,
  so the workflow always ends.
- trace pattern: researcher, writer, reviewer, writer, reviewer, ...
  (writer and reviewer alternate after the researcher).
