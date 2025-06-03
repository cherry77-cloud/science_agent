from typing import Optional, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME

from prompts import (
    PAPER_AGENT_SYSTEM_INTRO,
    PAPER_FIRST_ITERATION_TASK,
    PAPER_REVISION_TASK,
    PAPER_AGENT_INSTRUCTIONS,
    PAPER_REWARD_AGENT_SYSTEM,
    PAPER_REWARD_AGENT_INSTRUCTIONS
)


class LatexOutput(BaseModel):
    latex_code: str
    explanation: Optional[str] = (
        "Brief explanation of the changes made or the overall structure."
    )


class PaperScore(BaseModel):
    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="A numerical score from 1 (poor) to 10 (excellent).",
    )
    feedback: str = Field(
        ...,
        description="Specific, actionable feedback for improvement. If compilation failed, focus on LaTeX errors. If successful, focus on content, structure, and adherence to the plan.",
    )
    compilation_fixed: Optional[bool] = Field(
        None,
        description="Set to true if previous compilation failed and you believe your feedback addresses the LaTeX errors.",
    )
    positive_aspects: Optional[str] = Field(
        None, description="1-2 strengths of the current paper version."
    )
    key_issues_to_address: Optional[List[str]] = Field(
        None, description="2-3 most critical issues for revision."
    )


def get_paper_agent(iteration: int = 0, model_id: str = OPENAI_MODEL_NAME) -> Agent:
    if iteration == 0:
        task_specific_instructions = PAPER_FIRST_ITERATION_TASK
    else:
        task_specific_instructions = PAPER_REVISION_TASK.format(iteration=iteration + 1)

    return Agent(
        name=f"PaperAgent_Iter{iteration}",
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        system_message=PAPER_AGENT_SYSTEM_INTRO + task_specific_instructions,
        response_model=LatexOutput,
        structured_outputs=True,
        instructions=PAPER_AGENT_INSTRUCTIONS,
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
    )


def get_reward_agent(model_id: str = OPENAI_MODEL_NAME) -> Agent:
    return Agent(
        name="RewardAgent",
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        system_message=PAPER_REWARD_AGENT_SYSTEM,
        response_model=PaperScore,
        structured_outputs=True,
        instructions=PAPER_REWARD_AGENT_INSTRUCTIONS,
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
    )
