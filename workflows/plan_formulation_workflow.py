from typing import Optional, List, Dict, Any
import re
from agno.workflow import Workflow, RunResponse
from agno.agent import Agent
from agno.run.response import RunEvent
from agno.utils.log import logger
from pydantic import BaseModel, Field
from utils.text_processing import extract_prompt


class PlanFormulationWorkflow(Workflow):
    phd: Agent
    postdoc: Agent
    max_steps: int

    # Define a Pydantic model for the structured output of this workflow
    class PlanFormulationOutput(BaseModel):
        final_plan: Optional[str] = None
        dialogue_history: List[str]
        status_message: str
        last_agent_response_content: Optional[str] = (
            None,
        )  # Store content of last agent interaction
        need_data: Optional[bool] = Field(
            True,
            description="True if the formulated plan requires data for execution/validation, False otherwise.",
        )
        need_experiment: Optional[bool] = Field(
            True,
            description="True if the formulated plan requires running code experiments, False otherwise.",
        )

    def __init__(
        self,
        phd_agent: Agent,
        postdoc_agent: Agent,
        dataset_dir=None,
        max_steps: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)  # Pass kwargs to parent Workflow
        self.phd = phd_agent
        self.postdoc = postdoc_agent
        self.max_steps = max_steps
        self.prev_comm_phd = "No previous command."  # Initialize prev_comm for PhD
        self.prev_comm_postdoc = (
            "No previous command."  # Initialize prev_comm for Postdoc
        )
        self.dataset_dir = dataset_dir

    def run(
        self,
        topic: str,
        lit_review_sum: str,
        initial_dialogue_history: Optional[List[str]] = None,
    ) -> RunResponse:
        """
        Orchestrates the plan formulation dialogue between PhD and Postdoc agents.
        """
        logger.info(f"PlanFormulationWorkflow started for topic: {topic}")
        # Yield initial status immediately for streaming if called by another workflow
        # (though in this case it returns a single RunResponse at the end)

        dialogue_history = (
            initial_dialogue_history if initial_dialogue_history is not None else []
        )

        last_successful_agent_content: Optional[str] = (
            None  # To store the content of the truly last agent response
        )

        for step in range(self.max_steps):
            current_step_log = f"Plan formulation step {step + 1}/{self.max_steps}"
            logger.info(current_step_log)
            dialogue_history_str = "\n".join(dialogue_history)

            # 1. Postdoc Agent
            logger.info("Calling Postdoc Agent.")
            postdoc_prompt = f"""Current Literature Review: {lit_review_sum}
Task instructions: {self.postdoc.phase_prompt('plan formulation')}\n{self.postdoc.command_descriptions('plan formulation')}
[Objective] Your goal is to perform research on the following topic: {topic}
Dialogue History (Postdoc last spoke):
{dialogue_history_str}
Your previous command was: {self.prev_comm_postdoc}.
PhD Student just said: {dialogue_history[-1] if dialogue_history and dialogue_history[-1].startswith("PhD Student:") else "Nothing yet, you start."}
Make sure your new output is very different if repeating. Please produce a single command (DIALOGUE or PLAN) below:
"""

            try:
                postdoc_agent_response: RunResponse = self.postdoc.run(
                    # ... (existing postdoc_prompt construction) ...
                    postdoc_prompt,  # Your existing prompt for postdoc
                    stream=False,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                last_successful_agent_content = (
                    postdoc_agent_response.content
                    if isinstance(postdoc_agent_response.content, str)
                    else str(postdoc_agent_response.content)
                )
                self.prev_comm_postdoc = last_successful_agent_content

                postdoc_response_text = (
                    postdoc_agent_response.content
                    if isinstance(postdoc_agent_response.content, str)
                    else str(postdoc_agent_response.content)
                )

                if "```PLAN" in postdoc_response_text:
                    raw_plan_with_metadata = extract_prompt(
                        postdoc_response_text, "PLAN"
                    ).strip()
                    logger.info(
                        f"Postdoc submitted PLAN block: {raw_plan_with_metadata}"
                    )

                    plan_text, need_data, need_experiment = self._parse_plan_content(
                        raw_plan_with_metadata
                    )

                    if need_data is None or need_experiment is None:
                        logger.warning(
                            "Postdoc PLAN did not include 'Requires Data' or 'Requires Experiment' flags correctly. Assuming False for missing flags."
                        )
                        # Provide feedback for the next round if you were to continue or handle error
                        dialogue_history.append(
                            "Postdoc: [PLAN submitted but missing required flags. Please ensure 'Requires Data: True/False' and 'Requires Experiment: True/False' are included.]"
                        )
                        # If you want to force a re-try, you could 'continue' here.
                        # For now, we'll proceed with defaults if missing.
                        need_data = need_data if need_data is not None else False
                        need_experiment = (
                            need_experiment if need_experiment is not None else False
                        )

                    output = self.PlanFormulationOutput(
                        final_plan=plan_text,
                        dialogue_history=dialogue_history,
                        status_message="Plan submitted by Postdoc.",
                        last_agent_response_content=last_successful_agent_content,
                        need_data=need_data,
                        need_experiment=need_experiment,
                    )
                    return RunResponse(
                        content=output, event=RunEvent.workflow_completed
                    )
                elif "```DIALOGUE" in postdoc_response_text:
                    postdoc_dialogue = extract_prompt(
                        postdoc_response_text, "DIALOGUE"
                    ).strip()
                    logger.info("Postdoc says:" + postdoc_dialogue)
                    dialogue_history.append(f"Postdoc: {postdoc_dialogue}")
                else:
                    logger.warning(
                        "Postdoc Agent did not use ```DIALOGUE``` or ```PLAN``` command."
                    )
                    dialogue_history.append(
                        "Postdoc: [Did not provide a valid DIALOGUE or PLAN command. Please guide the PhD student or submit a PLAN with Requires Data/Experiment flags.]"
                    )

            except Exception as e:
                logger.error(f"Error during Postdoc Agent run in step {step + 1}: {e}")
                dialogue_history.append(f"Postdoc: [Error in response: {e}]")

            dialogue_history_str = "\n".join(dialogue_history)

            # 2. PhD Agent
            logger.info("Calling PhD Student Agent.")
            phd_prompt = f"""Current Literature Review: {lit_review_sum}
Task instructions: {self.phd.phase_prompt('plan formulation')}\n{self.phd.command_descriptions('plan formulation')}
[Objective] Your goal is to perform research on the following topic: {topic}
Dialogue History (PhD last spoke or is starting):
{dialogue_history_str}
Your previous command was: {self.prev_comm_phd}.
Postdoc just said: {dialogue_history[-1] if dialogue_history and dialogue_history[-1].startswith("Postdoc:") else "No new input from Postdoc, continue based on your understanding."}
Make sure your new output is very different if repeating. Please produce a single DIALOGUE command below:
"""  # PhD only produces DIALOGUE, Postdoc produces PLAN
            try:
                phd_agent_response: RunResponse = self.phd.run(
                    phd_prompt,
                    stream=False,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                last_successful_agent_content = (
                    phd_agent_response.content
                    if isinstance(phd_agent_response.content, str)
                    else str(phd_agent_response.content)
                )
                self.prev_comm_phd = last_successful_agent_content

                phd_response_text = (
                    phd_agent_response.content
                    if isinstance(phd_agent_response.content, str)
                    else str(phd_agent_response.content)
                )

                if "```DIALOGUE" in phd_response_text:
                    phd_dialogue = extract_prompt(phd_response_text, "DIALOGUE").strip()
                    logger.info("PhD says:" + phd_dialogue)
                    dialogue_history.append(f"PhD Student: {phd_dialogue}")
                else:
                    logger.warning("PhD Agent did not use ```DIALOGUE``` command.")
                    dialogue_history.append(
                        "PhD Student: [Did not provide a DIALOGUE command. Please continue the discussion with the Postdoc.]"
                    )
            except Exception as e:
                logger.error(f"Error during PhD Agent run in step {step + 1}: {e}")
                dialogue_history.append(f"PhD Student: [Error in response: {e}]")

        logger.warning(
            f"Plan formulation phase reached maximum steps ({self.max_steps}) without a final PLAN command."
        )
        output = self.PlanFormulationOutput(
            final_plan="Max steps reached. No final plan submitted.",
            dialogue_history=dialogue_history,
            status_message=f"Max steps ({self.max_steps}) reached during plan formulation.",
            last_agent_response_content=last_successful_agent_content,
            need_data=False,  # Default if max steps reached without plan
            need_experiment=False,  # Default
        )
        return RunResponse(content=output, event=RunEvent.workflow_completed)

    def _parse_plan_content(
        self, raw_plan_content: str
    ) -> tuple[str, Optional[bool], Optional[bool]]:
        """
        Parses the raw plan content from the Postdoc agent to extract the plan text,
        need_data, and need_experiment flags.
        Handles optional markdown bolding around the flag keywords.
        """
        plan_text = raw_plan_content
        need_data = None
        need_experiment = None

        # Regex to find "Requires Data: True/False", allowing for optional markdown bolding
        # (?:\*\*?) matches an optional "**"
        data_match = re.search(
            r"(?:\*\*?)?Requires Data:(?:\*\*?)?\s*(True|False)",
            raw_plan_content,
            re.IGNORECASE,
        )
        if data_match:
            need_data_str = data_match.group(1).lower()  # Group 1 is (True|False)
            need_data = need_data_str == "true"
            # Remove the matched line (including potential markdown) from the plan_text
            plan_text = plan_text.replace(data_match.group(0), "").strip()
        else:
            logger.warning(
                f"Could not find 'Requires Data' flag in PLAN content: {raw_plan_content[:200]}..."
            )

        # Regex to find "Requires Experiment: True/False", allowing for optional markdown bolding
        experiment_match = re.search(
            r"(?:\*\*?)?Requires Experiment:(?:\*\*?)?\s*(True|False)",
            raw_plan_content,
            re.IGNORECASE,
        )
        if experiment_match:
            need_experiment_str = experiment_match.group(
                1
            ).lower()  # Group 1 is (True|False)
            need_experiment = need_experiment_str == "true"
            # Remove the matched line (including potential markdown) from the plan_text
            plan_text = plan_text.replace(experiment_match.group(0), "").strip()
        else:
            logger.warning(
                f"Could not find 'Requires Experiment' flag in PLAN content: {raw_plan_content[:200]}..."
            )

        # Remove any trailing newlines that might have been left after removing metadata lines
        plan_text = plan_text.strip()

        return plan_text, need_data, need_experiment
