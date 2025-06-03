from agno.agent import Agent
from agno.models.openai import OpenAIChat
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME

from prompts import (
    PHD_LITERATURE_REVIEW_PROMPT,
    PHD_PLAN_FORMULATION_PROMPT, 
    PHD_RESULTS_INTERPRETATION_PROMPT,
    PHD_LITERATURE_REVIEW_COMMANDS,
    PHD_PLAN_FORMULATION_COMMANDS,
    PHD_RESULTS_INTERPRETATION_COMMANDS,
    POSTDOC_PLAN_FORMULATION_PROMPT,
    POSTDOC_RESULTS_INTERPRETATION_PROMPT,
    POSTDOC_PLAN_FORMULATION_COMMANDS,
    POSTDOC_RESULTS_INTERPRETATION_COMMANDS,
    ML_ENGINEER_DATA_PREPARATION_PROMPT,
    ML_ENGINEER_DATA_PREPARATION_COMMANDS,
    SW_ENGINEER_DATA_PREPARATION_PROMPT,
    SW_ENGINEER_DATA_PREPARATION_COMMANDS
)


class PhDAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def phase_prompt(self, phase):
        phase_str = ""
        if phase == "literature review":
            phase_str = PHD_LITERATURE_REVIEW_PROMPT
        elif phase == "plan formulation":
            phase_str = PHD_PLAN_FORMULATION_PROMPT
        elif phase == "results interpretation":
            phase_str = PHD_RESULTS_INTERPRETATION_PROMPT
        return phase_str

    def command_descriptions(self, phase):
        if phase == "literature review":
            return PHD_LITERATURE_REVIEW_COMMANDS
        elif phase == "plan formulation":
            return PHD_PLAN_FORMULATION_COMMANDS
        elif phase == "results interpretation":
            return PHD_RESULTS_INTERPRETATION_COMMANDS


class PostDocAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def phase_prompt(self, phase):
        phase_str = ""
        if phase == "plan formulation":
            phase_str = POSTDOC_PLAN_FORMULATION_PROMPT
        elif phase == "results interpretation":
            phase_str = POSTDOC_RESULTS_INTERPRETATION_PROMPT
        return phase_str

    def command_descriptions(self, phase):
        if phase == "plan formulation":
            return POSTDOC_PLAN_FORMULATION_COMMANDS
        elif phase == "results interpretation":
            return POSTDOC_RESULTS_INTERPRETATION_COMMANDS


class MLEngineerAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def phase_prompt(self, phase):
        phase_str = ""
        if phase == "data preparation":
            phase_str = ML_ENGINEER_DATA_PREPARATION_PROMPT
        return phase_str

    def command_descriptions(self, phase):
        if phase == "data preparation":
            return ML_ENGINEER_DATA_PREPARATION_COMMANDS
        return ()


class SWEngineerAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def phase_prompt(self, phase):
        phase_str = ""
        if phase == "data preparation":
            phase_str = SW_ENGINEER_DATA_PREPARATION_PROMPT
        return phase_str

    def command_descriptions(self, phase):
        if phase == "data preparation":
            return SW_ENGINEER_DATA_PREPARATION_COMMANDS
        return ""


def get_laboratory_agent(api_key=OPENAI_API_KEY):
    phd = PhDAgent(
        model=OpenAIChat(
            id=OPENAI_MODEL_NAME, api_key=api_key, base_url=OPENAI_BASE_URL
        ),
        description="You are a PhD student at a well-known university",
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
        add_history_to_messages=True,
        num_history_runs=5,
    )
    postdoc = PostDocAgent(
        model=OpenAIChat(
            id=OPENAI_MODEL_NAME, api_key=api_key, base_url=OPENAI_BASE_URL
        ),
        description="a computer science postdoctoral student at a top university.",
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
        add_history_to_messages=True,
        num_history_runs=5,
    )
    swe = SWEngineerAgent(
        model=OpenAIChat(
            id=OPENAI_MODEL_NAME, api_key=api_key, base_url=OPENAI_BASE_URL
        ),
        description="a computer science postdoctoral student at a top university.",
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
        add_history_to_messages=True,
        num_history_runs=5,
    )
    mle = MLEngineerAgent(
        model=OpenAIChat(
            id=OPENAI_MODEL_NAME, api_key=api_key, base_url=OPENAI_BASE_URL
        ),
        description="a computer science postdoctoral student at a top university.",
        retries=5,
        delay_between_retries=5,
        exponential_backoff=True,
        add_history_to_messages=True,
        num_history_runs=5,
    )
    return phd, postdoc, swe, mle
