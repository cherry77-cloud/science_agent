from typing import List, Dict, Any, Iterator, Optional
from pathlib import Path
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.media import ImageArtifact
from agno.agent import Agent
from agno.utils.log import logger

from agents.data_agents import (
    get_data_preparation_agent,
    get_data_analysis_agent,
    get_dataset_summary_agent,
    DataPreparationDecision,
    DataAnalysisResult,
    SingleDatasetFinalSummary,
)
from utils.load_and_save_dataset import load_and_save_toy_dataset
from tools.code_execution import execute_python_code_simple
from config import DATA_PLOT_OUTPUT_DIR, GENERATED_DATA_DIR


class DataAnalysisWorkflow(Workflow):
    data_analyzer: Agent
    data_preparer: Agent
    summarizer: Agent
    max_steps_per_dataset: int

    def __init__(self, max_steps_per_dataset: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.data_analyzer = get_data_analysis_agent()
        self.data_preparer = get_data_preparation_agent()
        self.summarizer = get_dataset_summary_agent()
        self.max_steps_per_dataset = max_steps_per_dataset
        DATA_PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _run_single_dataset_react_loop(
        self, dataset_path_str: str
    ) -> List[Dict[str, Any]]:
        dataset_step_results_log: List[Dict[str, Any]] = []
        agent_context_for_this_dataset = f"Initial dataset path: {dataset_path_str}\n"
        dataset_name_for_plots = (
            Path(dataset_path_str).stem.replace(" ", "_").lower()
        )  # For plot filenames

        for step_num in range(self.max_steps_per_dataset):
            # Construct the prompt for the agent, including the dynamic OUTPUT_DIR
            # and the reminder about nearing max steps.

            prompt_for_agent = agent_context_for_this_dataset.replace(
                "{DATA_PLOT_OUTPUT_DIR}", str(DATA_PLOT_OUTPUT_DIR.resolve())
            )
            prompt_for_agent = prompt_for_agent.replace(
                "{GENERATED_DATA_DIR}", str(GENERATED_DATA_DIR.resolve())
            )

            progress_percentage = int((step_num + 1) / self.max_steps_per_dataset * 100)
            prompt_for_agent = prompt_for_agent.replace(
                "{X_PERCENT_OF_MAX_STEPS}%", f"{progress_percentage}%"
            )

            # Ensure the prompt for the agent also knows dataset_name_for_plots if it's used in filename generation
            prompt_for_agent += f"\nUse '{dataset_name_for_plots}' as a base for generating plot filenames.\n"

            yield {
                "status": "info",
                "message": f"Dataset: {Path(dataset_path_str).name} - Step {step_num + 1}/{self.max_steps_per_dataset}",
                "detail": f"Prompting DataAnalyzer with context:\n{prompt_for_agent}...",
            }

            agent_run_response: RunResponse = self.data_analyzer.run(
                message=prompt_for_agent
            )

            if not agent_run_response.content or not isinstance(
                agent_run_response.content, DataAnalysisResult
            ):
                error_message = f"Error: DataAnalyzerAgent did not return a valid DataAnalysisStepResult for {Path(dataset_path_str).name} at step {step_num + 1}. Content: {str(agent_run_response.content)}"
                logger.error(error_message)
                dataset_step_results_log.append(
                    {"error": error_message, "step": step_num + 1}
                )
                yield {"status": "error", "message": error_message}
                break

            current_step_result: DataAnalysisResult = agent_run_response.content

            yield {
                "status": "agent_step_result",
                "dataset": Path(dataset_path_str).name,
                "step": step_num + 1,
                "agent_output": current_step_result.model_dump(),
            }

            # Execute the generated code
            stdout, stderr = execute_python_code_simple(
                current_step_result.generated_code,
                dataset_path_str,
                target_output_dir=DATA_PLOT_OUTPUT_DIR,
            )

            dataset_step_results_log.append(
                {
                    "step_description": current_step_result.current_step_description,
                    "generated_code": current_step_result.generated_code,
                    "next_step_thought": current_step_result.next_step_thought,
                    "execution_stdout": stdout,
                    "execution_stderr": stderr,
                    "step_findings_summary": current_step_result.summary_of_this_step_findings,
                    "plot_filenames": current_step_result.plot_filenames_generated_in_this_step
                    or [],
                    "next_step_thought_by_agent": current_step_result.next_step_thought,
                    "is_final_analysis_step": current_step_result.is_final_analysis_step,  # Store this
                }
            )

            yield {
                "status": "code_execution_result",
                "dataset": Path(dataset_path_str).name,
                "step": step_num + 1,
                "stdout": stdout,
                "stderr": stderr,
                "plot_filenames": current_step_result.plot_filenames_generated_in_this_step,
            }

            # Add reported plot files as artifacts
            if current_step_result.plot_filenames_generated_in_this_step:
                for (
                    plot_file_path_str
                ) in current_step_result.plot_filenames_generated_in_this_step:
                    plot_file_path = Path(plot_file_path_str)
                    if plot_file_path.exists() and plot_file_path.is_file():
                        artifact = ImageArtifact(
                            filepath=plot_file_path, id=plot_file_path.name
                        )
                        if self.images is None:
                            self.images = []
                        self.images.append(artifact)
                    else:
                        logger.warning(
                            f"Agent reported plot file '{plot_file_path_str}' but it was not found or is not a file."
                        )

            if current_step_result.is_final_analysis_step:
                yield {
                    "status": "info",
                    "message": f"Analysis for {Path(dataset_path_str).name} marked as final by agent.",
                }
                break

            # Prepare context for the next iteration
            agent_context_for_this_dataset = (
                f"Dataset: {dataset_path_str}\n"
                f"You are on step {step_num + 2} of {self.max_steps_per_dataset} for this dataset.\n"
                f"--- Context from previous step ({step_num + 1}) ---\n"
                f"Code Executed:\n```python\n{current_step_result.generated_code}\n```\n"
                f"Execution STDOUT:\n{stdout}\n"
                f"Execution STDERR:\n{stderr}\n"
                f"Agent's Summary of Previous Step Findings: {current_step_result.summary_of_this_step_findings}\n"
                f"Agent's Thought for Next Step (from previous #TODO): {current_step_result.next_step_thought}\n\n"
                f"Now, determine and perform the next logical analysis/preprocessing step."
            )

        # After loop (either break or max_steps reached)
        if step_num == self.max_steps_per_dataset - 1 and (
            not hasattr(current_step_result, "is_final_analysis_step")
            or not current_step_result.is_final_analysis_step
        ):  # type: ignore
            yield {
                "status": "warning",
                "message": f"Max steps ({self.max_steps_per_dataset}) reached for {Path(dataset_path_str).name}. Analysis might be incomplete.",
            }
            # Add a final entry to the log indicating max steps reached if not marked final
            if not any(
                "error" in step for step in dataset_step_results_log
            ):  # Only if no prior errors
                dataset_step_results_log.append(
                    {
                        "step_description": "Max steps reached",
                        "generated_code": "# No further code generated due to max steps.",
                        "code_explanation": "Maximum iteration steps reached for this dataset.",
                        "execution_stdout": "",
                        "execution_stderr": "",
                        "step_findings_summary": "Analysis halted due to reaching the maximum allowed steps.",
                        "plot_filenames": [],
                        "next_step_thought_by_agent": "Further analysis was stopped due to step limits.",
                        "is_final_analysis_step": True,  # Mark as final to stop further processing for this dataset
                    }
                )

        return dataset_step_results_log

    # Main `run` method needs to be adapted to the new structure of `_run_single_dataset_react_loop`
    # if it now yields dicts instead of RunResponse directly for intermediate statuses.
    # The example above for `_run_single_dataset_react_loop` shows it yielding dicts.
    def run(
        self,
        plan: str,
        topic: Optional[str] = None,
        dataset_paths: Optional[List[str]] = None,
    ) -> Iterator[RunResponse]:
        final_reports_per_dataset: List[Dict[str, Any]] = []
        actual_dataset_paths_to_process: List[str] = []

        # Step 1: Determine data source
        if dataset_paths and all(Path(p).exists() for p in dataset_paths):
            yield RunResponse(
                content={
                    "status": "info",
                    "message": f"Using user-provided datasets: {dataset_paths}",
                },
                event=RunEvent.workflow_started,
            )
            actual_dataset_paths_to_process = dataset_paths
        else:
            yield RunResponse(
                content={
                    "status": "info",
                    "message": "User did not provide data. Attempting to find or simulate data...",
                },
                event=RunEvent.workflow_started,
            )

            prep_agent_prompt = (
                f"Analysis Plan: {plan}\n"
                f"Topic: {topic if topic else 'General'}\n"
                f"Available sklearn toy datasets and their descriptions:\n"
                f"- load_diabetes(): Diabetes patient data, 10 features, regression target.\n"
                f"- load_digits(): 8x8 handwritten digit images, 64 features, classification target.\n"
                f"- load_breast_cancer(): Tumor features, 30 features, binary classification target.\n"
                f"- load_linnerud(): Athlete exercise and physiological data, multi-output regression.\n"
                f"- load_wine(): Wine chemical analysis, 13 features, multi-class classification target.\n\n"
                f"Please decide on the data source and provide your response as a JSON object matching the DataPreparationDecision model."
            )

            prep_decision_response: RunResponse = self.data_preparer.run(
                message=prep_agent_prompt
            )

            if not prep_decision_response.content or not isinstance(
                prep_decision_response.content, DataPreparationDecision
            ):
                error_msg = f"DataPreparerAgent failed to return a valid decision. Content: {str(prep_decision_response.content)[:500]}"
                logger.error(error_msg)
                yield RunResponse(
                    content={"status": "error", "message": error_msg},
                    event=RunEvent.run_error,
                )
                return  # End workflow

            decision: DataPreparationDecision = prep_decision_response.content
            yield RunResponse(
                content={
                    "status": "info",
                    "message": f"Data preparation decision: {decision.action}",
                    "reasoning": decision.reasoning,
                },
                event=RunEvent.run_response,
            )

            if (
                decision.action == "use_toy_dataset"
                and decision.selected_toy_dataset_name
            ):
                try:
                    toy_path = load_and_save_toy_dataset(
                        decision.selected_toy_dataset_name
                    )
                    actual_dataset_paths_to_process = [str(toy_path)]
                    yield RunResponse(
                        content={
                            "status": "info",
                            "message": f"Loaded and saved toy dataset '{decision.selected_toy_dataset_name}' to {toy_path}",
                        },
                        event=RunEvent.run_response,
                    )
                except Exception as e:
                    error_msg = f"Failed to load toy dataset '{decision.selected_toy_dataset_name}': {e}"
                    logger.error(error_msg)
                    yield RunResponse(
                        content={"status": "error", "message": error_msg},
                        event=RunEvent.run_error,
                    )
                    return
            elif (
                decision.action == "generate_simulated_data"
                and decision.simulation_code_to_generate_data
            ):
                yield RunResponse(
                    content={
                        "status": "info",
                        "message": "Executing simulation code...",
                    },
                    event=RunEvent.run_response,
                )
                GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

                stdout, stderr = execute_python_code_simple(
                    decision.simulation_code_to_generate_data,
                    target_output_dir=GENERATED_DATA_DIR,
                )

                yield RunResponse(
                    content={
                        "status": "code_execution_result",
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                    event=RunEvent.run_response,
                )

                if stderr:
                    error_msg = f"Error executing simulation code: {stderr}"
                    logger.error(error_msg)
                    yield RunResponse(
                        content={"status": "error", "message": error_msg},
                        event=RunEvent.run_error,
                    )
                    return

                # Extract file paths from stdout (assuming the code prints them, one per line)
                simulated_paths = [
                    line.strip()
                    for line in stdout.splitlines()
                    if Path(line.strip()).exists()
                ]
                if not simulated_paths:
                    error_msg = (
                        "Simulation code did not print valid file paths to stdout."
                    )
                    logger.error(error_msg)
                    yield RunResponse(
                        content={"status": "error", "message": error_msg},
                        event=RunEvent.run_error,
                    )
                    return
                actual_dataset_paths_to_process = simulated_paths
                yield RunResponse(
                    content={
                        "status": "info",
                        "message": f"Generated simulated data at: {simulated_paths}",
                    },
                    event=RunEvent.run_response,
                )

            else:
                error_msg = "Invalid decision from DataPreparerAgent."
                logger.error(error_msg)
                yield RunResponse(
                    content={"status": "error", "message": error_msg},
                    event=RunEvent.run_error,
                )
                return

        # --- Step 2: Iteratively analyze each determined dataset path ---
        # (This part reuses your _run_single_dataset_react_loop and subsequent summarization per dataset)
        # ... (rest of your existing run method from the previous good version) ...
        # Ensure it iterates over `actual_dataset_paths_to_process`

        # Example of how to integrate the loop:
        for i, path_str in enumerate(actual_dataset_paths_to_process):
            yield RunResponse(
                content={
                    "status": "info",
                    "message": f"Starting iterative analysis for dataset {i + 1}/{len(actual_dataset_paths_to_process)}: {Path(path_str).name}",
                },
                event=RunEvent.run_response,
            )

            single_dataset_full_log: List[Dict[str, Any]] = []
            for step_update_dict in self._run_single_dataset_react_loop(
                path_str
            ):  # This is your existing ReAct loop
                single_dataset_full_log.append(step_update_dict)
                yield RunResponse(content=step_update_dict, event=RunEvent.run_response)

            # (The summarizer logic from your previous good version would go here for this dataset)
            yield RunResponse(
                content={
                    "status": "info",
                    "message": f"Completed iterative analysis for {Path(path_str).name}. Now summarizing...",
                },
                event=RunEvent.run_response,
            )

            if not single_dataset_full_log or any(
                step.get("status") == "error" for step in single_dataset_full_log
            ):

                error_summary = f"Skipping summary for {Path(path_str).name} due to errors or empty log in analysis steps."
                logger.error(error_summary)
                final_reports_per_dataset.append(
                    {
                        "dataset_path": path_str,
                        "error": error_summary,
                        "analysis_log": single_dataset_full_log,
                    }
                )
                yield RunResponse(
                    content={"status": "error", "message": error_summary},
                    event=RunEvent.run_response,
                )
                continue

            summarizer_input_str = f"Please provide a final summary and consolidated preprocessing code for the dataset: {path_str}, based on the following analysis steps and their outputs:\n\n"

            for step_idx, step_log_entry in enumerate(single_dataset_full_log):
                if step_log_entry.get("status") == "agent_step_result":
                    agent_out = step_log_entry.get("agent_output", {})
                    summarizer_input_str += (
                        f"--- Agent Plan for Step {step_idx + 1} ---\n"
                    )
                    summarizer_input_str += f"Description: {agent_out.get('current_step_description', 'N/A')}\n"
                    summarizer_input_str += f"Generated Code:\n```python\n{agent_out.get('generated_code', '')}\n```\n"
                    summarizer_input_str += f"Next Step Thought: {agent_out.get('next_step_thought', 'N/A')}\n\n"
                elif step_log_entry.get("status") == "code_execution_result":
                    summarizer_input_str += (
                        f"--- Code Execution Result for Step {step_idx + 1} ---\n"
                    )
                    summarizer_input_str += (
                        f"STDOUT:\n{step_log_entry.get('stdout', '')}\n"
                    )
                    if step_log_entry.get("stderr"):
                        summarizer_input_str += (
                            f"STDERR:\n{step_log_entry.get('stderr', '')}\n"
                        )

            summary_agent_response: RunResponse = self.summarizer.run(
                message=summarizer_input_str
            )

            if summary_agent_response.content and isinstance(
                summary_agent_response.content, SingleDatasetFinalSummary
            ):
                dataset_final_summary: SingleDatasetFinalSummary = (
                    summary_agent_response.content
                )
                final_reports_per_dataset.append(dataset_final_summary.model_dump())
                yield RunResponse(
                    content={
                        "status": "info",
                        "message": f"Summary generated for {Path(path_str).name}.",
                    },
                    event=RunEvent.run_response,
                )
            else:
                # ... (handle summarizer error as before) ...
                error_summary = f"Failed to generate final summary for {Path(path_str).name}: {summary_agent_response.content}"
                logger.error(error_summary)
                final_reports_per_dataset.append(
                    {
                        "dataset_path": path_str,
                        "error": error_summary,
                        "analysis_log": single_dataset_full_log,
                    }
                )
                yield RunResponse(
                    content={"status": "error", "message": error_summary},
                    event=RunEvent.run_response,
                )

        # Final Workflow Output
        final_run_response = RunResponse(
            content=final_reports_per_dataset,
            content_type="List[SingleDatasetFinalSummary]",  # Or just "list"
            event=RunEvent.workflow_completed.value,
            run_id=self.run_id,
            session_id=self.session_id,
            workflow_id=self.workflow_id,
        )

        yield final_run_response
