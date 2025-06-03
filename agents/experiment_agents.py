from typing import Optional, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME

from prompts import (
    CODE_AGENT_ROLE,
    CODE_AGENT_GENERAL_TASK,
    CODE_GENERATION_RULES,
    FIGURE_GENERATION_GUIDELINES,
    INPUT_DATA_SUMMARY,
    OUTPUT_FORMAT_INSTRUCTIONS,
    FIRST_ITERATION_TASK,
    REVISION_TASK,
    CODE_AGENT_INSTRUCTIONS,
    REFLECTION_AGENT_DESCRIPTION,
    REFLECTION_AGENT_INSTRUCTIONS,
    CODE_REWARD_AGENT_DESCRIPTION,
    CODE_REWARD_AGENT_INSTRUCTIONS
)


class ExperimentCodeOutput(BaseModel):
    experiment_code: str = Field(
        ...,
        description="The complete Python code for the experiment. This code should be directly runnable.",
    )
    explanation: Optional[str] = Field(
        "Brief explanation of the code's logic or changes made.",
        description="A brief explanation of the code or changes.",
    )
    focused_task: Optional[str] = Field(
        None,
        description="The primary task or part of the plan this code iteration addresses.",
    )


class ReflectionOutput(BaseModel):
    reflection_content: str = Field(
        ...,
        description="Detailed reflection on the code's execution. Identify issues, suggest improvements, and assess if the code aligns with the plan and produces meaningful results.",
    )
    suggestions_for_next_iteration: Optional[List[str]] = Field(
        None,
        description="Specific, actionable suggestions for the CodeAgent's next iteration.",
    )
    code_is_runnable: bool = Field(
        description="True if the code ran without Python execution errors, False otherwise (based on running_signal)."
    )
    output_looks_meaningful: bool = Field(
        description="True if the captured output seems to make sense in the context of the plan, False otherwise."
    )


class CodeScore(BaseModel):
    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="A numerical score from 1 (poor) to 10 (excellent) based on correctness, adherence to plan, and quality of results/reflection.",
    )
    feedback: str = Field(
        ...,
        description="Overall feedback summarizing why this score was given, considering the plan, code, execution, and reflection.",
    )
    code_quality_rating: Optional[int] = Field(
        None, ge=1, le=5, description="Rating for code quality (1-5)."
    )
    results_alignment_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Rating for how well results align with the plan (1-5).",
    )
    reflection_usefulness_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Rating for the usefulness of the reflection (1-5).",
    )


def get_code_agent(
    model_id: str = OPENAI_MODEL_NAME,
    iteration=0,
    save_loc_for_figures: str = "figures",
) -> Agent:
    code_generation_rules = CODE_GENERATION_RULES
    figure_generation_guidelines = FIGURE_GENERATION_GUIDELINES.format(save_loc_for_figures=save_loc_for_figures)
    
    if iteration == 0:
        task_specific_instructions = f"""
{FIRST_ITERATION_TASK}
{OUTPUT_FORMAT_INSTRUCTIONS}
"""
    else:
        task_specific_instructions = f"""
{REVISION_TASK}
{OUTPUT_FORMAT_INSTRUCTIONS}
"""

    final_description = f"{CODE_AGENT_ROLE}\n\n{CODE_AGENT_GENERAL_TASK}\n\n{code_generation_rules}\n\n{figure_generation_guidelines}\n\n{INPUT_DATA_SUMMARY}\n\n{task_specific_instructions}"

    return Agent(
        name=f"CodeAgent_Iter{iteration}",
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        description=final_description,
        response_model=ExperimentCodeOutput,
        structured_outputs=True,
        instructions=CODE_AGENT_INSTRUCTIONS,
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
    )


def get_reflection_agent(model_id: str = OPENAI_MODEL_NAME) -> Agent:
    return Agent(
        name="ReflectionAgent",
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        description=REFLECTION_AGENT_DESCRIPTION,
        response_model=ReflectionOutput,
        structured_outputs=True,
        instructions=REFLECTION_AGENT_INSTRUCTIONS,
        retries=5,
        delay_between_retries=10,
        exponential_backoff=True,
    )


def get_code_reward_agent(
    model_id: str = OPENAI_MODEL_NAME,
) -> Agent:
    return Agent(
        name="CodeRewardAgent",
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        description=CODE_REWARD_AGENT_DESCRIPTION,
        response_model=CodeScore,
        structured_outputs=True,
        instructions=CODE_REWARD_AGENT_INSTRUCTIONS,
        retries=3,
        delay_between_retries=5,
        exponential_backoff=True,
    )
