#  0. Importing the necessary libraries
from agents import Agent, Runner, RunContextWrapper, function_tool, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from dataclasses import dataclass

import os
from dotenv import load_dotenv, find_dotenv

# 0.1. Loading the environment variables
load_dotenv(find_dotenv())

set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # 🔑 Get your API key from environment
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

external_client: AsyncOpenAI = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)


model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-3.5-flash-lite", openai_client=external_client
    )

@dataclass
class CalculateInfo:
    length: str
    width: str
    uid: int

# @function_tool
# async def fetch_user_age(wrapper: RunContextWrapper[CalculateInfo]) -> str:
#  """Returns the age of the user."""
#  return f"User {wrapper.context.name} is 47 years old"

@function_tool
def calculate_area(wrapper: RunContextWrapper[CalculateInfo]) -> str:
    """Calculate the area of a rectangle."""
    length = wrapper.context.length
    width =  wrapper.context.width
    area = length * width
    return f"Area = {length} × {width} = {area} square units"

def main():
    """Learn Model Settings with simple examples."""
    question = "Calculate Area"
    calculate_info = CalculateInfo(length=4, width= 3 , uid=123)
    
    # agent_cold = Agent[CalculateInfo](
    #     name="Cold Agent",
    #     instructions="You are a helpful assistant.",
    #     model_settings=ModelSettings(temperature=0.1),
    #     model=model
    # )

    agent_required = Agent[CalculateInfo](
        name="Required",
        tools=[calculate_area],
        model_settings=ModelSettings(tool_choice="required"),
        model=model
    )

    print("Auto Tool Choice:")
    result_auto = Runner.run_sync(agent_required, question, context=calculate_info)
    print(result_auto.final_output)

if __name__ == "__main__":
    main()