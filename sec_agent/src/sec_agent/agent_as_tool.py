from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, trace
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



spanish = Agent(name='Spanish Translator', 
               instructions="Translate what the user says into Spanish. Only output Spanish.",
               model=model)

summarizer = Agent(
 name="Summarizer",
 instructions="Summarize the given text in 2 short bullet points.",
 model=model,
)

coach = Agent(
 name="Writing Coach",
 instructions=("You help users improve messages.\n"
 "- If they ask for Spanish, call translate_to_spanish.\n"
 "- If they ask for a summary, call summarize_text.\n"
 "- Otherwise, give a short tip yourself."),
 model=model,
 tools=[
    spanish.as_tool(
        tool_name="translate_to_spanish",
        tool_description="Translate user text to spanish"
    ),
    summarizer.as_tool(
        tool_name="summarize_text",
        tool_description="Summarize text in 2 bullets"
    )
 ]
)
async def main():
    result = await Runner.run(coach, "Please translate to Spanish: Hello.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

