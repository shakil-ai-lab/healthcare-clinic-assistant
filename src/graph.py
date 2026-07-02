from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.agent import get_agent


# Define the graph state
class GraphState(TypedDict):
    question: str
    answer: str


# Load agent
agent = get_agent()


# -----------------------------
# Node 1
# -----------------------------
def receive_question(state: GraphState):

    print("Receiving user question...")

    return state


# -----------------------------
# Node 2
# -----------------------------
def agent_executor(state: GraphState):

    question = state["question"]

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    answer = response["messages"][-1].content

    return {
        "answer": answer
    }


# -----------------------------
# Node 3
# -----------------------------
def save_memory(state: GraphState):

    # Placeholder
    # Later we'll integrate LangGraph memory here

    print("Saving conversation...")

    return state


# -----------------------------
# Node 4
# -----------------------------
def final_answer(state: GraphState):

    print("Returning final answer...")

    return state


# Build Graph
builder = StateGraph(GraphState)

builder.add_node("receive_question", receive_question)
builder.add_node("agent_executor", agent_executor)
builder.add_node("save_memory", save_memory)
builder.add_node("final_answer", final_answer)

builder.add_edge(START, "receive_question")
builder.add_edge("receive_question", "agent_executor")
builder.add_edge("agent_executor", "save_memory")
builder.add_edge("save_memory", "final_answer")
builder.add_edge("final_answer", END)

graph = builder.compile()


if __name__ == "__main__":

    result = graph.invoke(
        {
            "question": "How can I book an appointment?"
        }
    )

    print("\nAssistant:\n")

    print(result["answer"])