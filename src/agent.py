from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from src.tools import clinic_policy_search

from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")


# Initialize HF LLM
model = ChatHuggingFace(llm=llm)

# List of tools
tools = [clinic_policy_search]

# Create Agent
agent = create_react_agent(
    model=model,
    tools=tools
)


def get_agent():
    """
    Return the initialized agent.
    """
    return agent

if __name__ == "__main__":

    agent = get_agent()

    question = "How can I book an appointment?"

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

    print(response["messages"][-1].content)
