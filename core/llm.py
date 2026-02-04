from langchain_google_genai import ChatGoogleGenerativeAI
from AI_make_Problems.models.respon import ProgrammingProblem, ProgrammingCritic
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7
).with_structured_output(ProgrammingProblem)

critic_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
).with_structured_output(ProgrammingCritic)