from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from agno.agent import Agent
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME
from agno.models.openai import OpenAIChat

from prompts import (
    DATA_PREPARATION_DECISION_SYSTEM,
    DATA_ANALYSIS_INSTRUCTIONS,
    DATASET_SUMMARY_INSTRUCTIONS
)


class DataPreparationDecision(BaseModel):
    action: Literal["use_toy_dataset", "generate_simulated_data"]
    selected_toy_dataset_name: Optional[str] = Field(
        None,
        description="If action is 'use_toy_dataset', the name of the sklearn.datasets function to call (e.g., 'load_diabetes').",
    )
    simulation_code_to_generate_data: Optional[str] = Field(
        None,
        description="If action is 'generate_simulated_data', the Python code to create and save simulated dataframes as local files. This code MUST print the list of absolute paths to the saved file(s).",
    )
    reasoning: str = Field(description="The reasoning behind the chosen action.")


class DataAnalysisResult(BaseModel):
    dataset_path: str = Field(description="Path of the dataset being analyzed.")
    current_step_description: str = Field(
        description="A brief description of what this specific analysis step accomplishes."
    )
    generated_code: str = Field(
        description=(
            "Python code generated for *this single analysis/preprocessing step*. "
            "This code MUST include detailed comments explaining each part. "
            "The code MUST end with a comment '# TODO: [Next logical action description]' indicating what the next code block should achieve. "
            "If generating a plot, ensure the code also prints descriptive statistics or textual summaries related to the plot's content."
        )
    )
    next_step_thought: str = Field(
        description=(
            "Based on the # TODO comment in your generated_code and the nature of the current step, "
            "provide a clear thought process or question for the *next* analysis step. "
            "For example, if code checked for missing values, this might be 'Analyze the types of missing values and decide on an imputation strategy.'"
        )
    )
    plot_filenames_generated_in_this_step: Optional[List[str]] = Field(
        None,
        description="List of *full paths* to plot filenames generated and saved by the Python code in *this current step*.",
    )
    is_final_analysis_step: bool = Field(
        description="Set to true if you believe all necessary analysis and preprocessing for this single dataset is complete and this was the final meaningful step. Otherwise, set to false."
    )
    summary_of_this_step_findings: str = Field(
        description="A concise summary of what was learned or achieved in this specific code execution step. If a plot was generated, briefly describe what the plot shows based on any textual output from the code."
    )


class SingleDatasetFinalSummary(BaseModel):
    dataset_path: str = Field(description="Path of the dataset that was analyzed.")
    comprehensive_analysis_summary: str = Field(
        description="A detailed summary of all findings, observations, and insights gathered from all analysis steps for this single dataset."
    )
    consolidated_preprocessing_code_for_dataset: str = Field(
        description="A single, clean Python script combining all *preprocessing* code steps generated for this dataset. This script should be runnable and take the raw dataset as input to produce a cleaned/preprocessed version."
    )
    key_plots_description: Optional[str] = Field(
        None,
        description="A textual description summarizing what the key generated plots for this dataset visually represent. Refer to the plot filenames if available.",
    )


def get_data_preparation_agent(model_id: str = OPENAI_MODEL_NAME) -> Agent:
    """
    Create a Data Preparation Agent that can decide between using
    toy datasets or generating simulated data.
    """
    return Agent(
        name="DataPreparationAgent",
        description="Agent responsible for choosing appropriate datasets for analysis.",
        system_message=DATA_PREPARATION_DECISION_SYSTEM,
        response_model=DataPreparationDecision,
        structured_outputs=True,
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
    )


def get_data_analysis_agent(model_id: str = OPENAI_MODEL_NAME) -> Agent:
    return Agent(
        name="IncrementalDataAnalyzer",
        description="An AI agent that incrementally analyzes a dataset by generating and explaining single Python code steps, and plans the next step.",
        instructions=DATA_ANALYSIS_INSTRUCTIONS,
        response_model=DataAnalysisResult,
        structured_outputs=True,
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
    )


def get_dataset_summary_agent(model_id: str = OPENAI_MODEL_NAME) -> Agent:
    return Agent(
        name="DatasetReportSummarizer",
        description="An AI agent that takes a series of analysis step results for a single dataset and produces a final summary and consolidated preprocessing code for that dataset.",
        instructions=DATASET_SUMMARY_INSTRUCTIONS,
        response_model=SingleDatasetFinalSummary,
        structured_outputs=True,
        model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
    )
