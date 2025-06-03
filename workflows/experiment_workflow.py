import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, List
from dataclasses import field, dataclass

from agno.agent import Agent
from agno.workflow import Workflow, RunResponse
from agno.run.response import RunEvent
from agno.utils.log import logger, log_info

from tools.code_execution import execute_code
from agents.experiment_agents import (
    get_code_agent,
    get_reflection_agent,
    get_code_reward_agent,
    ExperimentCodeOutput,
    ReflectionOutput,
    CodeScore,
)


@dataclass
class IterationArtefacts:
    iteration: int
    score: float
    code: str
    execution_result: Dict[str, Any]
    reflection: Optional[ReflectionOutput]
    reward_feedback: Optional[str]
    focused_task: Optional[str]
    temp_figures_path: Optional[Path] = None
    final_code_file_path: Optional[Path] = None
    final_result_file_path: Optional[Path] = None
    final_figures_dir_path: Optional[Path] = None


class ExperimentCodingWorkflow(Workflow):
    description: str = (
        "A workflow to iteratively write and refine experimental Python code, saving only the best result."
    )

    code_agent_initial: Agent = field(init=False)
    code_agent_revise: Agent = field(init=False)
    reflection_agent: Agent = field(init=False)
    reward_agent: Agent = field(init=False)

    def __init__(
        self,
        max_iterations: int = 10,
        min_score_to_accept: float = 8.0,
        output_dir_name: str = r"C:/project/Agentlab_demo",
        name: Optional[str] = "ExperimentCodingWorkflow_Instance",
        description: Optional[
            str
        ] = "Generates experimental Python code iteratively, saving the best.",
        debug_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name, description=description, debug_mode=debug_mode, **kwargs
        )
        self.max_iterations = max_iterations
        self.min_score_to_accept = min_score_to_accept
        self.base_output_dir = Path(output_dir_name)
        self.best_code_output_dir = self.base_output_dir / "best_code"
        self.global_figures_output_dir = self.base_output_dir / "figures"

        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.best_code_output_dir.mkdir(parents=True, exist_ok=True)
        self.global_figures_output_dir.mkdir(parents=True, exist_ok=True)

        _placeholder_fig_path = str(self.base_output_dir / "_temp_figs_placeholder")
        self.code_agent_initial = get_code_agent(
            iteration=0, save_loc_for_figures=_placeholder_fig_path
        )
        self.code_agent_revise = get_code_agent(
            iteration=1, save_loc_for_figures=_placeholder_fig_path
        )
        self.reflection_agent = get_reflection_agent()
        self.reward_agent = get_code_reward_agent()

    def _save_best_artefacts(self, best_result: IterationArtefacts):
        """Helper to save code, execution results for the single best iteration, and copy figures."""
        logger.info(
            f"Saving artefacts for best iteration {best_result.iteration} (Score: {best_result.score})."
        )

        for f_py in self.best_code_output_dir.glob("*.py"):
            f_py.unlink()
        code_file_name = f"best_experiment_code_iter_{best_result.iteration}_score_{best_result.score:.2f}.py"
        best_result.final_code_file_path = self.best_code_output_dir / code_file_name
        with open(best_result.final_code_file_path, "w", encoding="utf-8") as f:
            f.write(best_result.code)
        log_info(f"Saved best code to: {best_result.final_code_file_path}")

        for f_json in self.best_code_output_dir.glob("*.json"):
            f_json.unlink()
        result_file_name = f"best_execution_result_iter_{best_result.iteration}_score_{best_result.score:.2f}.json"
        best_result.final_result_file_path = (
            self.best_code_output_dir / result_file_name
        )
        with open(best_result.final_result_file_path, "w", encoding="utf-8") as f:
            json.dump(best_result.execution_result, f, indent=2)
        log_info(
            f"Saved best execution result to: {best_result.final_result_file_path}"
        )

        best_result.final_figures_dir_path = self.global_figures_output_dir
        if best_result.temp_figures_path and best_result.temp_figures_path.exists():
            for f_img in self.global_figures_output_dir.glob("*.*"):
                f_img.unlink()
            copied_count = 0
            for item in best_result.temp_figures_path.iterdir():
                if item.is_file():
                    try:
                        shutil.copy2(item, self.global_figures_output_dir / item.name)
                        copied_count += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to copy figure {item.name} to {self.global_figures_output_dir}: {e}"
                        )
            if copied_count > 0:
                log_info(
                    f"Copied {copied_count} figure(s) from {best_result.temp_figures_path} to {self.global_figures_output_dir}"
                )
            else:
                log_info(
                    f"No figures found or copied from {best_result.temp_figures_path}"
                )
        else:
            log_info(
                f"No temporary figures path specified or found for the best iteration: {best_result.temp_figures_path}"
            )

    def run(
        self,
        topic: str,
        lit_review_sum: str,
        plan: str,
        data_preparation_result: List,
        stream_intermediate_steps: bool = False,
    ) -> Iterator[RunResponse]:

        best_result_so_far: Optional[IterationArtefacts] = None

        current_code: Optional[str] = None
        current_execution_result: Optional[Dict[str, Any]] = None
        current_reflection: Optional[ReflectionOutput] = None
        current_reward_feedback: Optional[str] = None
        current_score: float = (
            -1.0
        )  # Initialize to a value lower than any possible score
        focused_task: Optional[str] = None

        initial_prompt_data = f"""
Topic:
{topic}

Literature Review Summary (for context):
{lit_review_sum}

Research Plan:
{plan}

Datasets Access and Their Analysis Result (the codes will be prepended to your generated experiment code):
{data_preparation_result}
"""

        for i in range(self.max_iterations):
            iteration_num = i + 1
            iteration_message = (
                f"--- Code Iteration {iteration_num}/{self.max_iterations} ---"
            )
            logger.info(iteration_message)
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"\n\n**{iteration_message}**\n",
                )

            # --- Define a *temporary* directory for THIS iteration's figures ---
            # CodeAgent will save figures here. If this iter becomes the best, figures are copied.
            temp_iter_figures_dir = self.base_output_dir / f"tmp_figs"
            temp_iter_figures_dir.mkdir(parents=True, exist_ok=True)

            # 1. Code Agent generates/revises code
            # Pass the specific *temporary* figure path for this iteration to the agent
            # The agent's prompt needs to instruct it to save figures to this exact path.
            if i == 0:
                self.code_agent_initial = get_code_agent(
                    iteration=0, save_loc_for_figures=str(temp_iter_figures_dir)
                )
                agent_to_use = self.code_agent_initial
                code_agent_prompt = f"Based on the following information, please generate the first version of the experimental Python code. Ensure all necessary imports are included. Save any generated figures to the directory: '{str(temp_iter_figures_dir)}'.\n{initial_prompt_data}"
            else:
                self.code_agent_revise = get_code_agent(
                    iteration=i, save_loc_for_figures=str(temp_iter_figures_dir)
                )
                agent_to_use = self.code_agent_revise
                code_agent_prompt = f"""
Please revise the Python experimental code. Ensure all necessary imports are included.
Save any newly generated or modified figures to the directory: '{str(temp_iter_figures_dir)}'.

Original Requirements:
{initial_prompt_data}

Previous Code:
```python
{current_code}
```

Previous Execution Result:
{json.dumps(current_execution_result, indent=2) if current_execution_result else "N/A"}

Reflection & Suggestions from Previous Iteration:
{current_reflection.reflection_content if current_reflection else "N/A"}
Suggestions: {current_reflection.suggestions_for_next_iteration if current_reflection and current_reflection.suggestions_for_next_iteration else "N/A"}

Feedback from Evaluator (Score: {current_score if i > 0 else 'N/A'}):
{current_reward_feedback if current_reward_feedback else "N/A"}
"""
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"🖋️ Code Agent (iteration {iteration_num}) working... Figures will be temporarily saved to '{temp_iter_figures_dir}'.\n",
                )

            code_agent_response: RunResponse = agent_to_use.run(
                code_agent_prompt, stream=False
            )
            if not isinstance(code_agent_response.content, ExperimentCodeOutput):
                error_msg = "Code Agent did not return the expected ExperimentCodeOutput object."
                logger.error(error_msg)
                if stream_intermediate_steps:
                    yield RunResponse(event=RunEvent.run_error, content=error_msg)
                break
            generated_part = re.sub(
                r"print\((['\"])\n\s*(.*?)\1\)",
                r"print(\1\2\1)",
                code_agent_response.content.experiment_code,
            )
            dataset_code = "\n".join(
                [
                    d["consolidated_preprocessing_code_for_dataset"]
                    for d in data_preparation_result
                ]
            )
            current_code = dataset_code.strip() + "\n" + generated_part.strip()

            generation_explanation = code_agent_response.content.explanation
            focused_task = code_agent_response.content.focused_task

            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"💻 Code Agent (iteration {iteration_num}) generated code for task '{focused_task}'. Explanation: {generation_explanation}\n```python\n{current_code}\n```\n",
                )

            # 2. Execute the code
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"🚀 Executing code (iteration {iteration_num})...\n",
                )
            current_execution_result = execute_code(code_str=current_code)

            if stream_intermediate_steps:
                stdout_snippet = current_execution_result.get("stdout", "")
                stderr_snippet = current_execution_result.get("stderr", "")
                exit_c = current_execution_result.get("exit_code", "N/A")
                exec_output_summary = f"Exit Code: {exit_c}\nStdout: {stdout_snippet}\nStderr: {stderr_snippet}"
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"⚙️ Execution result (iteration {iteration_num}):\n{exec_output_summary}\n",
                )

            # 3. Reflection Agent analyzes
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"🤔 Reflection Agent (iteration {iteration_num}) analyzing...\n",
                )
            reflection_prompt = f"""
Research Plan:
{plan}
Executed Code:
```python
{current_code}
```
Execution Result:
{json.dumps(current_execution_result, indent=2)}
Please reflect on this execution and provide suggestions. Consider if figures were saved to '{str(temp_iter_figures_dir)}' as instructed.
"""
            reflection_agent_response: RunResponse = self.reflection_agent.run(
                reflection_prompt, stream=False
            )
            if not isinstance(reflection_agent_response.content, ReflectionOutput):
                error_msg = "Reflection Agent did not return ReflectionOutput."
                logger.error(error_msg)
                current_reflection = ReflectionOutput(
                    reflection_content="Error in reflection.",
                    suggestions_for_next_iteration=[],
                    code_is_runnable=False,
                    output_looks_meaningful=False,
                )
            else:
                current_reflection = reflection_agent_response.content

            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"💡 Reflection (iteration {iteration_num}): {current_reflection.reflection_content}\nSuggestions: {current_reflection.suggestions_for_next_iteration}\nRunnable: {current_reflection.code_is_runnable}, Meaningful Output: {current_reflection.output_looks_meaningful}\n",
                )

            # 4. Reward Agent scores
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"⚖️ Reward Agent (iteration {iteration_num}) scoring...\n",
                )
            reward_prompt = f"""
Research Plan:
{plan}
Latest Experiment Code:
```python
{current_code}
```
Execution Result of Latest Code:
{json.dumps(current_execution_result, indent=2)}
Reflection Agent's Analysis:
{current_reflection.reflection_content}
Reflection Agent's Suggestions: {current_reflection.suggestions_for_next_iteration}
Score the current state. Consider if figures were appropriately generated and saved to the temp dir '{str(temp_iter_figures_dir)}'.
"""
            reward_agent_response: RunResponse = self.reward_agent.run(
                reward_prompt, stream=False
            )
            if not isinstance(reward_agent_response.content, CodeScore):
                error_msg = "Reward Agent did not return CodeScore."
                logger.error(error_msg)
                current_score = 0.0
                current_reward_feedback = "Error in scoring."
            else:
                current_score = reward_agent_response.content.score
                current_reward_feedback = reward_agent_response.content.feedback

            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"📊 Reward Agent (iteration {iteration_num}) score: {current_score}/10. Feedback: {current_reward_feedback}\n",
                )

            # --- Update Best Result So Far ---
            if current_reflection.code_is_runnable:  # Basic check
                if (
                    best_result_so_far is None
                    or current_score >= best_result_so_far.score
                ):
                    logger.info(
                        f"New best score: {current_score} at iteration {iteration_num}. Previous best: {best_result_so_far.score if best_result_so_far else 'None'}."
                    )
                    best_result_so_far = IterationArtefacts(
                        iteration=iteration_num,
                        score=current_score,
                        code=current_code,
                        execution_result=current_execution_result,
                        reflection=current_reflection,
                        reward_feedback=current_reward_feedback,
                        focused_task=focused_task,
                        temp_figures_path=temp_iter_figures_dir,  # Store path where agent *should* have saved figures
                    )
                    # Save artefacts for this new best result immediately
                    self._save_best_artefacts(best_result_so_far)
                    if stream_intermediate_steps:
                        yield RunResponse(
                            event=RunEvent.run_response,
                            content=f"🏆 New best result! Score: {current_score}/10. Artefacts updated in '{self.best_code_output_dir}' and figures in '{self.global_figures_output_dir}'.\n",
                        )

            # Check for stopping condition
            if (
                current_score >= self.min_score_to_accept
                and current_reflection.code_is_runnable
                and current_reflection.output_looks_meaningful
            ):
                logger.info(
                    f"Score {current_score} meets acceptance threshold. Stopping."
                )
                if stream_intermediate_steps:
                    yield RunResponse(
                        event=RunEvent.run_response,
                        content="✅ Score meets acceptance threshold. Finalizing.\n",
                    )
                break

            if i == self.max_iterations - 1:
                logger.info("Max iterations reached.")
                if stream_intermediate_steps:
                    yield RunResponse(
                        event=RunEvent.run_response,
                        content="🏁 Max iterations reached.\n",
                    )

            try:
                shutil.rmtree(temp_iter_figures_dir)
                logger.debug(
                    f"Cleaned up temporary figure directory: {temp_iter_figures_dir}"
                )
            except OSError as e:
                logger.warning(
                    f"Could not remove temporary figure directory {temp_iter_figures_dir}: {e}"
                )

        if best_result_so_far:
            final_summary_message = (
                f"Experiment coding workflow completed. Best result (from Iteration {best_result_so_far.iteration}):\n"
                f"Score: {best_result_so_far.score}/10\n"
                f"Code saved at: {best_result_so_far.final_code_file_path.resolve() if best_result_so_far.final_code_file_path else 'N/A'}\n"
                f"Execution Result saved at: {best_result_so_far.final_result_file_path.resolve() if best_result_so_far.final_result_file_path else 'N/A'}\n"
                f"Figures saved in directory: {best_result_so_far.final_figures_dir_path.resolve() if best_result_so_far.final_figures_dir_path else 'N/A'}\n"
                f"Focused Task: {best_result_so_far.focused_task}\n"
                f"Feedback: {best_result_so_far.reward_feedback}\n"
                f"Execution Stdout (snippet): {best_result_so_far.execution_result.get('stdout', '')}\n"
                f"Execution Stderr (snippet): {best_result_so_far.execution_result.get('stderr', '')}\n"
            )
            extra_data_payload = {
                "best_iteration_details": {
                    "iteration": best_result_so_far.iteration,
                    "score": best_result_so_far.score,
                    "code_file_path": (
                        str(best_result_so_far.final_code_file_path.resolve())
                        if best_result_so_far.final_code_file_path
                        else None
                    ),
                    "result_file_path": (
                        str(best_result_so_far.final_result_file_path.resolve())
                        if best_result_so_far.final_result_file_path
                        else None
                    ),
                    "figures_dir_path": (
                        str(best_result_so_far.final_figures_dir_path.resolve())
                        if best_result_so_far.final_figures_dir_path
                        else None
                    ),
                    "code": best_result_so_far.code,
                    "execution_result": best_result_so_far.execution_result,
                    "focused_task": best_result_so_far.focused_task,
                    "reward_feedback": best_result_so_far.reward_feedback,
                }
            }
            final_run_response = RunResponse(
                event=RunEvent.run_completed,
                content=final_summary_message,
                extra_data=extra_data_payload,
            )
            yield final_run_response
        else:
            yield RunResponse(
                event=RunEvent.run_error,
                content="No usable experiment code was generated or executed successfully enough to be saved as best.",
            )
