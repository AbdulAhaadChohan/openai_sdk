
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, trace,handoff
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


fitness_coach = Agent(
 name="Fitness Coach",
 instructions=(
 "You're a running coach. Ask 1-2 quick questions, then give a week plan. "
 "Keep it simple and encouraging. No medical advice."
 ),
 model=model,
)
study_coach = Agent(
 name="Study Coach",
 instructions=(
 "You're a study planner. Ask for the current routine, then give a "
 "one-week schedule. Keep steps small and doable."
 ),
 model=model,
)
router = Agent(
 name="Coach Router",
 instructions=(
 "Route the user:\n"
 "- running, workout, stamina -> hand off to the Fitness Coach.\n"
 "- exams, study plan, focus, notes -> hand off to the Study Coach.\n"
 "After the handoff the specialist continues the conversation."
 ),
 model=model,
 handoffs=[study_coach, handoff(fitness_coach)],
)
async def main():
 result = await Runner.run(router, "I want to run 5km in 8 weeks. Can you help?")
 print(result.final_output)
 print("answered by:", result.last_agent.name)

if __name__ == "__main__":
    asyncio.run(main())