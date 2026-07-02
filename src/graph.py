from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent import get_agent


# ==========================================================
# Graph State
# ==========================================================

class GraphState(TypedDict):
    question: str
    answer: str


# ==========================================================
# Load Agent
# ==========================================================

agent = get_agent()


# ==========================================================
# Node 1 : Receive Question
# ==========================================================

def receive_question(state: GraphState) -> GraphState:
    """
    Receive the user's question.
    """

    print("\nReceiving user question...\n")

    return state


# ==========================================================
# Node 2 : Execute Agent
# ==========================================================

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


# ==========================================================
# Node 3 : Save Memory
# ==========================================================

def save_memory(state: GraphState) -> GraphState:
    """
    Placeholder node.

    Conversation state is automatically saved by
    LangGraph's MemorySaver checkpointer.
    """

    print("Conversation saved.\n")

    return state


# ==========================================================
# Node 4 : Final Answer
# ==========================================================

def final_answer(state: GraphState) -> GraphState:

    print("Returning final answer...\n")

    return state


# ==========================================================
# Build Graph
# ==========================================================

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


# ==========================================================
# Memory
# ==========================================================

memory = MemorySaver()


# ==========================================================
# Compile Graph
# ==========================================================

graph = builder.compile(
    checkpointer=memory
)


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "patient_001"
        }
    }

    while True:

        question = input("\nYou : ")

        if question.lower() in ["exit", "quit"]:
            break

        result = graph.invoke(
            {
                "question": question
            },
            config=config
        )

        print("\nAssistant :")
        print(result["answer"])