from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
import asyncio 
import os


load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # 🔑 Get your API key from environment
# GEMINI_API_KEY = os.getenv("OPENAI_API_KEY")  # 🔑 Get your API key from environment
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

external_client: AsyncOpenAI = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)


model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-3.5-flash-lite", openai_client=external_client
    )


agent1 = Agent(name='agent1', 
               instructions="What is the information of the user",
               model=model)
agent2 = Agent(name='agent2', 
               instructions="What is the information of the user",
               model=model)

async def main():
    first = await Runner.run(starting_agent=agent1, input="What is the information of the user")
    second = await Runner.run(starting_agent=agent2, input="What is the information of the user")

    print(first.final_output)
    print(second.final_output)


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    asyncio.run(main())