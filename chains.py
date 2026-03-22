"""LCEL chains: diet plan, fitness plan, and follow-up Q&A."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from prompts import diet_prompt, fitness_prompt, qa_prompt


def build_diet_chain(llm: BaseChatModel) -> Runnable[dict[str, Any], str]:
    return diet_prompt | llm | StrOutputParser()


def build_fitness_chain(llm: BaseChatModel) -> Runnable[dict[str, Any], str]:
    return fitness_prompt | llm | StrOutputParser()


def build_qa_chain(llm: BaseChatModel) -> Runnable[dict[str, Any], str]:
    return qa_prompt | llm | StrOutputParser()
