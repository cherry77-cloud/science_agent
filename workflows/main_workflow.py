import json
from typing import Optional

from agno.agent import Agent
from agno.workflow import Workflow
from agno.tools.arxiv import ArxivTools
from agno.utils.log import logger, log_info, log_debug
from agno.utils.common import is_empty
from agno.run.response import RunResponse, RunEvent

from agents.laboratory_agents import get_laboratory_agent
from utils.text_processing import extract_prompt
from utils.api_clients import query_model

from .experiment_workflow import ExperimentCodingWorkflow
from .plan_formulation_workflow import PlanFormulationWorkflow
from .data_preparation_workflow import DataAnalysisWorkflow
from .paper_writing_workflow import PaperWritingWorkflow


class AgentLab(Workflow):
    def __init__(
        self,
        research_topic,
        lab_dir,
        dataset_dir=None,
        paper_dir=None,
        note=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lab_dir = lab_dir
        self.phd, self.postdoc, self.swe, self.mle = get_laboratory_agent()
        self.max_full_text_length = 50000
        self.arxiv_paper_exp_time = 3
        self.prev_comm = ""
        self.max_steps = 10
        self.arxiv_tool = ArxivTools()
        self.phase = [
            ("literature review", self._literature_review),
            ("plan formulation", self._plan_formulation),
            ("data preparation", self._data_preparation),
            ("running experiments", self._run_experiments),
            ("results interpretation", self._result_interpretation),
            ("paper writing", self._paper_writing),
        ]
        self.session_state.setdefault("research_topic", research_topic)
        self.session_state.setdefault("lit_review", [])
        self.session_state.setdefault("full_text", "")
        self.session_state.setdefault("lit_review_sum", "")
        self.session_state.setdefault("plan", "")
        self.session_state.setdefault("phase_status", {})
        self.session_state.setdefault("data_preparation_result", [])
        self.session_state.setdefault("note", note)
        self.session_state.setdefault("dataset_dir", dataset_dir)
        self.session_state.setdefault("paper_dir", paper_dir)
        self.session_state.setdefault("need_data", True)
        self.session_state.setdefault("need_experiment", True)

    def run(self, **kwargs) -> Optional[RunResponse]:

        last_agent_response = None
        skip_to_phase: Optional[str] = None

        for phase_name, phase_method in self.phase:
            if skip_to_phase and phase_name != skip_to_phase:
                logger.info(
                    f"Skipping phase '{phase_name}' due to jump to '{skip_to_phase}'."
                )
                self.session_state.get("phase_status", {})[phase_name] = True
                continue

            if skip_to_phase and phase_name == skip_to_phase:
                skip_to_phase = None

            logger.info(f"Starting workflow phase: {phase_name}")

            if self.session_state.get("phase_status", {}).get(phase_name):
                logger.info(
                    f"Phase '{phase_name}' already marked as complete. Skipping."
                )
                continue

            logger.info(f"Entering Phase: {phase_name}")

            try:
                phase_last_response = phase_method(
                    self.session_state.get("research_topic")
                )

                if phase_last_response:
                    last_agent_response = phase_last_response

                if phase_name == "plan formulation":

                    needs_data = self.session_state.get("need_data", True)
                    needs_experiment = self.session_state.get("need_experiment", True)

                    logger.info(
                        f"After plan formulation: need_data={needs_data}, need_experiment={needs_experiment}"
                    )

                    if not needs_data:
                        if needs_experiment:
                            logger.info(
                                "Data not needed, but experiment is needed. Skipping to 'running experiments'."
                            )
                            skip_to_phase = "running experiments"
                            self.session_state.get("phase_status", {})[
                                "data preparation"
                            ] = True
                        else:
                            logger.info(
                                "Neither data nor experiment needed. Skipping to 'paper writing'."
                            )
                            skip_to_phase = "paper writing"
                            self.session_state.get("phase_status", {})[
                                "data preparation"
                            ] = True
                            self.session_state.get("phase_status", {})[
                                "running experiments"
                            ] = True
                            self.session_state.get("phase_status", {})[
                                "results interpretation"
                            ] = True

            except Exception as e:
                logger.critical(
                    f"Workflow failed during phase '{phase_name}' due to error: {e}",
                    exc_info=True,
                )
                return RunResponse(
                    content=f"Workflow aborted during phase '{phase_name}'. Error: {e}",
                    event=RunEvent.run_error,
                    workflow_id=self.workflow_id,
                )

        logger.info("Workflow execution finished.")
        if last_agent_response is None:
            return RunResponse(
                content="Workflow completed with no agent interaction.",
                event=RunEvent.workflow_completed,
                workflow_id=self.workflow_id,
            )

        return last_agent_response

    def _literature_review(self, topic) -> Optional[RunResponse]:
        logger.info("Starting _literature_review_phase")
        current_step_feedback = ""

        last_agent_response_in_phase = None
        phase = "literature review"
        for step in range(self.max_steps):
            logger.info(f"Literature review step {step + 1}/{self.max_steps}")
            prompt = f"""Task instructions: {self.phd.phase_prompt(phase)}\n{self.phd.command_descriptions(phase)}
[Objective] Your goal is to perform research on the following topic:{topic}
Current Literature Review Papers Collected: {len(self.session_state.get('lit_review', []))}/20
{"Feedback from previous step:" + current_step_feedback if current_step_feedback else ""}
Your previous command was: {self.prev_comm}.
Make sure your new output is very different.\nPlease produce a single command below:\n"""

            try:
                agent_response = self.phd.run(
                    prompt,
                    stream=False,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                last_agent_response_in_phase = agent_response

                if agent_response and agent_response.event == RunEvent.run_error:
                    logger.error(
                        f"Agent run failed in step {step + 1}: {agent_response.content}"
                    )
                    feedback_for_next_step = f"[Workflow Feedback] Agent encountered an error: {agent_response.content}. Please review the last command and try again."
                elif agent_response and agent_response.event == RunEvent.run_cancelled:
                    logger.warning(f"Agent run was cancelled in step {step + 1}")
                    feedback_for_next_step = f"[Workflow Feedback] Agent run was cancelled. Please provide the next step."
                else:
                    resp_text = (
                        agent_response.content
                        if isinstance(agent_response.content, str)
                        else str(agent_response.content)
                    )
                    self.prev_comm = resp_text
                    feedback_for_next_step = ""
                    if "```SUMMARY" in resp_text:
                        query = extract_prompt(resp_text, "SUMMARY").strip()
                        logger.info(
                            f"Workflow detected SUMMARY command from Agent with query: {query}"
                        )
                        try:
                            summary_results_text = (
                                self.arxiv_tool.search_arxiv_and_return_articles(
                                    query=query
                                )
                            )
                            if summary_results_text and summary_results_text.strip():
                                feedback_for_next_step += f"You requested arXiv papers related to the query '{query}', here were the results:\n{summary_results_text}\n"
                                feedback_for_next_step += "Review these results carefully. If any look promising based on the titles/summaries, use the ```FULL_TEXT\narXiv paper ID\n``` command for a specific paper. If none look promising, use ```SUMMARY\n...\n``` again with a refined query, or use ```ADD_PAPER\n...\n``` if you already have relevant papers to add.\n"
                            else:
                                feedback_for_next_step += (
                                    f"Your search for '{query}' returned no results.\n"
                                )
                                feedback_for_next_step += "Please try using the ```SUMMARY\n...\n``` command with a different search query.\n"

                        except Exception as tool_error:
                            logger.error(
                                f"Error during ArxivTools SUMMARY search for query '{query}': {tool_error}"
                            )
                            feedback_for_next_step += f"[Workflow Feedback] Failed to perform SUMMARY search for '{query}' due to an error: {tool_error}\n"
                            feedback_for_next_step += "Please try using the ```SUMMARY\n...\n``` command again with the same or a different query.\n"

                    elif "```FULL_TEXT" in resp_text:
                        paper_id = extract_prompt(resp_text, "FULL_TEXT").strip()
                        logger.info(
                            f"Workflow detected FULL_TEXT command from Agent with ID: {paper_id}"
                        )
                        try:
                            read_results_json_str = self.arxiv_tool.read_arxiv_papers(
                                id_list=[paper_id], pages_to_read=None
                            )
                            read_results = json.loads(read_results_json_str)
                            full_text = ""
                            if (
                                read_results
                                and isinstance(read_results, list)
                                and len(read_results) > 0
                            ):
                                article_data = read_results[0]
                                if "content" in article_data and isinstance(
                                    article_data["content"], list
                                ):
                                    full_text = "\n".join(
                                        [
                                            page_content["text"]
                                            for page_content in article_data["content"]
                                            if "text" in page_content
                                            and page_content["text"]
                                        ]
                                    )
                                    logger.info(
                                        f"Successfully extracted text for {paper_id}."
                                    )
                                    if not full_text.strip():
                                        full_text = "[TEXT EXTRACTION FAILED]: Text content was empty."
                                        logger.warning(
                                            f"Extracted text for {paper_id} was empty."
                                        )
                                else:
                                    full_text = "[TEXT EXTRACTION FAILED]: Content structure missing or invalid."
                                    logger.warning(
                                        f"Could not find 'content' in read results for {paper_id}. Raw: {article_data.get('content', 'N/A')}"
                                    )
                            else:
                                full_text = "[TEXT EXTRACTION FAILED]: No results returned or invalid format for ID."
                                logger.warning(
                                    f"read_arxiv_papers returned empty or invalid results for {paper_id}. Raw: {read_results}"
                                )

                            max_len = self.max_full_text_length
                            clipped_full_text = full_text[:max_len]
                            if len(full_text) > max_len:
                                clipped_full_text += (
                                    "\n[Text clipped due to max length limit]"
                                )
                            self.session_state["full_text"] = (
                                paper_id,
                                clipped_full_text,
                            )
                            feedback_for_next_step += f"Full text for paper ID '{paper_id}':\n{clipped_full_text}\n"
                            feedback_for_next_step += f"Please read the full text provided. If paper ID '{paper_id}' is relevant to the research topic, use the ```ADD_PAPER\n{paper_id}\nYour Summary\n``` command to add it to the literature review, replacing 'Your Summary' with a concise summary. If it is not relevant, use ```FULL_TEXT\n...\n``` for another promising paper ID, or use ```SUMMARY\n...\n``` to find more papers.\n"

                        except Exception as tool_error:
                            logger.error(
                                f"Error during ArxivTools FULL_TEXT read for {paper_id}: {tool_error}"
                            )
                            feedback_for_next_step += f"[Workflow Feedback] Failed to get full text for '{paper_id}' due to an error: {tool_error}\n"
                            feedback_for_next_step += "Please try using the ```FULL_TEXT\n...\n``` command again with the same or a different paper ID, or use ```SUMMARY\n...\n``` to find more papers.\n"

                    elif "```ADD_PAPER" in resp_text:
                        add_paper_content = extract_prompt(
                            resp_text, "ADD_PAPER"
                        ).strip()
                        if not add_paper_content:
                            feedback_for_next_step += "[Workflow Feedback] You used the ADD_PAPER command but provided no content. Please provide the paper ID and a summary in the format: ID\\nSummary.\n"
                        else:
                            logger.info(
                                f"Workflow detected ADD_PAPER command from Agent with content: {add_paper_content}"
                            )
                            lines = add_paper_content.split("\n", 1)
                            if len(lines) == 2:
                                arxiv_id = lines[0].strip()
                                summary = lines[1].strip()
                                if arxiv_id and summary:
                                    if any(
                                        p["arxiv_id"] == arxiv_id
                                        for p in self.session_state["lit_review"]
                                    ):
                                        feedback_for_next_step += f"\n[Workflow Feedback] Paper {arxiv_id} was already added to the literature review. Please find new relevant papers.\n"
                                    else:
                                        if (
                                            self.session_state["full_text"][0]
                                            == arxiv_id
                                        ):
                                            paper_entry = {
                                                "arxiv_id": arxiv_id,
                                                "summary": summary,
                                                "full_text": self.session_state[
                                                    "full_text"
                                                ][1],
                                            }
                                        else:
                                            paper_entry = {
                                                "arxiv_id": arxiv_id,
                                                "summary": summary,
                                            }
                                        self.session_state["lit_review"].append(
                                            paper_entry
                                        )
                                        current_count = len(
                                            self.session_state["lit_review"]
                                        )
                                        feedback_for_next_step += f"\nSuccessfully added paper {arxiv_id} to literature review. Current count: {current_count}/10.\n"
                                        if current_count < 10:
                                            feedback_for_next_step += f"Goal is to collect 10 relevant papers. Please continue searching, reading, and adding papers. Current progress: {current_count}/10.\n"
                                        else:
                                            feedback_for_next_step += f"You have collected 10 relevant papers. The literature review phase is complete.\n"

                                else:
                                    feedback_for_next_step += "[Workflow Feedback] ADD_PAPER command detected but could not parse paper ID or summary. Please provide the content in the correct format (ID\\nSummary) after ```ADD_PAPER.\n"

                            else:
                                feedback_for_next_step += "[Workflow Feedback] ADD_PAPER command detected but content format is incorrect. Please use format: ID\\nSummary (Paper ID on first line, Summary on second).\n"

                    elif not is_empty(resp_text) and "```" not in resp_text:
                        feedback_for_next_step += f"[Workflow Feedback] You provided dialogue/text instead of a command. Please use the ```SUMMARY```, ```FULL_TEXT```, or ```ADD_PAPER``` commands to interact with the workflow for literature review. Last response: {resp_text[:100]}...\n"
                    else:
                        feedback_for_next_step += "[Workflow Feedback] Please provide a command (```SUMMARY```, ```FULL_TEXT```, or ```ADD_PAPER```) or dialogue in your response.\n"

            except Exception as e:
                logger.error(f"Error during Agent run in step {step + 1}: {e}")
                feedback_for_next_step = f"[Workflow Feedback] An error occurred during the Agent's run: {e}. Please try the step again or provide different instructions."

            current_step_feedback = feedback_for_next_step

            if len(self.session_state.get("lit_review", [])) >= 10:
                logger.info("Literature review completed (Reached 10 papers).")
                self.session_state["lit_review_sum"] = (
                    self._generate_lit_review_summary()
                )
                self.session_state["phase_status"]["literature review"] = True
                self.reset_agents()
                return last_agent_response_in_phase

            if step + 1 >= self.max_steps:
                logger.info(
                    f"Literature review completed (Max steps {self.max_steps} reached). Collected {len(self.session_state.get('lit_review', []))} papers."
                )
                self.session_state["lit_review_sum"] = (
                    self._generate_lit_review_summary()
                )
                self.session_state["phase_status"]["literature review"] = True
                self.reset_agents()
                return last_agent_response_in_phase

        logger.warning(
            "Literature review phase finished without explicitly returning or meeting conditions. Check phase logic."
        )
        self.session_state["phase_status"]["literature review"] = True
        self.reset_agents()
        return last_agent_response_in_phase

    def _plan_formulation(self, topic: str) -> Optional[RunResponse]:

        logger.info("Starting plan formulation phase using PlanFormulationWorkflow")

        lit_review_sum = self.session_state.get(
            "lit_review_sum", "No literature review summary available."
        )

        initial_dialogue = self.session_state.get("plan_dialogue_history", [])
        self.plan_formulation_workflow = PlanFormulationWorkflow(
            phd_agent=self.phd,
            postdoc_agent=self.postdoc,
            dataset_dir=self.session_state.get("paper_dir", None),
            max_steps=self.max_steps,
            name="PlanFormulationSubWorkflow",
            debug_mode=self.debug_mode,
        )
        plan_workflow_response: RunResponse = self.plan_formulation_workflow.run(
            topic=topic,
            lit_review_sum=lit_review_sum,
            initial_dialogue_history=initial_dialogue,
        )

        if (
            plan_workflow_response
            and plan_workflow_response.content
            and isinstance(
                plan_workflow_response.content,
                PlanFormulationWorkflow.PlanFormulationOutput,
            )
        ):
            output_data: PlanFormulationWorkflow.PlanFormulationOutput = (
                plan_workflow_response.content
            )

            logger.info(
                f"PlanFormulationWorkflow completed with status: {output_data.status_message}"
            )
            self.session_state["plan_dialogue_history"] = output_data.dialogue_history

            if (
                output_data.final_plan
                and output_data.final_plan
                != "Max steps reached. No final plan submitted."
            ):
                self.session_state["plan"] = output_data.final_plan
                self.session_state["phase_status"]["plan formulation"] = True
                self.session_state["need_data"] = output_data.need_data
                self.session_state["need_experiment"] = output_data.need_experiment
                self.session_state["plan_dialogue_history"] = []
            else:
                self.session_state["plan"] = output_data.final_plan
                self.session_state["phase_status"]["plan formulation"] = False
                logger.warning(f"Plan formulation ended: {output_data.status_message}")

            return RunResponse(
                content=f"Plan formulation result: {output_data.status_message}. Plan: {output_data.final_plan[:100]}...",
                event=RunEvent.run_response,
                workflow_id=self.workflow_id,
                session_id=self.session_id,
            )
        else:
            logger.error("PlanFormulationWorkflow did not return the expected output.")
            self.session_state["plan"] = "Error in plan formulation sub-workflow."
            self.session_state["phase_status"]["plan formulation"] = False
            return RunResponse(
                content="Error in plan formulation sub-workflow.",
                event=RunEvent.run_error,
            )

    def _data_preparation(self, topic) -> Optional[RunResponse]:
        logger.info("Starting data preparation phase")

        lit_review_sum = self.session_state.get(
            "lit_review_sum", "No literature review summary available."
        )
        plan = self.session_state.get("plan", "No plan available.")

        self.data_preparation_workflow = DataAnalysisWorkflow(
            name="Test Data Analysis Pipeline",
            description="Workflow to test data sourcing and iterative analysis.",
            max_steps_per_dataset=self.max_steps,
            debug_mode=False,
        )
        results_iterator = self.data_preparation_workflow.run_workflow(
            plan=plan,
            topic=topic,
            dataset_paths=self.session_state.get("dataset_dir", None),
        )

        final_workflow_output = None
        try:
            for response_chunk in results_iterator:
                if hasattr(response_chunk, "content") and response_chunk.content:
                    logger.debug(response_chunk.content)
                final_workflow_output = response_chunk
        except Exception as e:
            logger.error(f"An error occurred during workflow execution: {e}")
            return RunResponse(
                content=f"Error: Failed to instantiate ExperimentCodingWorkflow: {e}",
                event=RunEvent.run_error,
                workflow_id=self.workflow_id,
            )

        # --- 打印最终结果 ---
        if final_workflow_output and isinstance(final_workflow_output.content, list):
            logger.info("\n--- Final Reports Per Dataset ---")
            result = []
            for i, report in enumerate(final_workflow_output.content):
                logger.info(
                    f"\n--- Report for Dataset {i + 1} ({report.get('dataset_path', 'N/A')}) ---"
                )
                if "error" in report:
                    logger.error(f"Error: {report['error']}")
                    if "analysis_log" in report:
                        logger.info("Analysis Log (contains steps taken before error):")
                        for step_log_entry in report["analysis_log"]:
                            logger.info(
                                f"  Step Description: {step_log_entry.get('step_description', 'N/A')}"
                            )
                            logger.info(
                                f"  Agent Thought for Next: {step_log_entry.get('next_step_thought_by_agent', 'N/A')}"
                            )
                            # Optionally print code and output if needed for debugging errors
                else:
                    logger.info("\nConsolidated Preprocessing Code:")
                    logger.info(
                        report.get(
                            "consolidated_preprocessing_code_for_dataset",
                            "Not generated.",
                        )
                    )
                    logger.info("\nComprehensive Analysis Summary:")
                    logger.info(
                        report.get("comprehensive_analysis_summary", "Not generated.")
                    )
                    if report.get("key_plots_description"):
                        logger.info("\nKey Plots Description:")
                        logger.info(report.get("key_plots_description"))
                    result.append(report)
                    self.session_state["data_preparation_result"] = result

            return RunResponse(
                content=f"data preparation result:{result}",
                event=RunEvent.run_response,  # Or workflow_completed if this is the final step of _plan_formulation
                workflow_id=self.workflow_id,  # Main workflow's ID
                session_id=self.session_id,
            )

        else:
            logger.error("Workflow did not produce the expected list of final reports.")
            return RunResponse(
                content="Error in data preparation sub-workflow.",
                event=RunEvent.run_error,
            )

    def _run_experiments(self, topic) -> Optional[RunResponse]:
        logger.info("Starting _run_exp phase using ExperimentCodingWorkflow")

        # Get inputs from session_state
        lit_review_sum = self.session_state.get(
            "lit_review_sum", "No literature review summary available."
        )
        plan = self.session_state.get("plan", "No plan available.")
        data_preparation_result = self.session_state.get("data_preparation_result", [])

        # Define the output directory for the ExperimentCodingWorkflow.
        # This will be where "best_code" and "figures" subdirectories are created by the exp workflow.
        experiment_workflow_output_dir = self.lab_dir

        logger.info(
            f"Instantiating ExperimentCodingWorkflow. Output will be in: {experiment_workflow_output_dir}"
        )
        try:
            # Instantiate the new ExperimentCodingWorkflow
            exp_workflow = ExperimentCodingWorkflow(
                max_iterations=self.max_steps,
                min_score_to_accept=8.5,  # Or get from config/self
                output_dir_name=str(experiment_workflow_output_dir),  # Pass as string
                name=f"ExperimentSubWorkflow_{self.workflow_id or 'default'}",
                debug_mode=self.debug_mode,  # Or get from config/self
                # user_id and session_id are not directly used by ExperimentCodingWorkflow's __init__
                # but if your Agno Workflow base class handles them via **kwargs, they'll be passed.
                # Otherwise, if ExperimentCodingWorkflow needs them for its run_id, it gets it internally.
            )
            logger.info("ExperimentCodingWorkflow instantiated.")
        except Exception as e:
            logger.error(
                f"Failed to instantiate ExperimentCodingWorkflow: {e}", exc_info=True
            )
            self.session_state["phase_status"]["running experiments"] = False
            return RunResponse(
                content=f"Error: Failed to instantiate ExperimentCodingWorkflow: {e}",
                event=RunEvent.run_error,
                workflow_id=self.workflow_id,
            )

        logger.info("Running ExperimentCodingWorkflow...")
        final_exp_workflow_response = None
        try:
            # The run method of ExperimentCodingWorkflow is a generator
            for response_chunk in exp_workflow.run(
                topic=topic,
                lit_review_sum=lit_review_sum,
                plan=plan,
                data_preparation_result=data_preparation_result,
                stream_intermediate_steps=True,
            ):

                if hasattr(response_chunk, "content") and response_chunk.content:
                    logger.debug(response_chunk.content)
                final_result = response_chunk

        except Exception as e:
            logger.error(
                f"An error occurred while running ExperimentCodingWorkflow: {e}",
                exc_info=True,
            )
            self.session_state["phase_status"]["running experiments"] = False
            return RunResponse(
                content=f"Error during ExperimentWorkflow execution: {e}",
                event=RunEvent.run_error,
                workflow_id=self.workflow_id,
            )

        # --- Process the final response from ExperimentCodingWorkflow ---
        if final_exp_workflow_response.event == RunEvent.run_error:
            logger.error(
                f"ExperimentCodingWorkflow failed: {final_exp_workflow_response.content}"
            )
            self.session_state["phase_status"]["running experiments"] = False
            return final_exp_workflow_response  # Propagate the error response

        if final_exp_workflow_response.event != RunEvent.run_completed:
            logger.warning(
                f"ExperimentCodingWorkflow ended with status: {final_exp_workflow_response.event}. Content: {final_exp_workflow_response.content}"
            )
            # Depending on the event, you might still want to proceed or mark as error
            self.session_state["phase_status"][
                "running experiments"
            ] = False  # Assuming non-completed is an issue
            return final_exp_workflow_response

        # Successfully completed, now extract data from extra_data
        if (
            not final_exp_workflow_response.extra_data
            or "best_iteration_details" not in final_exp_workflow_response.extra_data
        ):
            error_msg = "ExperimentCodingWorkflow completed but did not return expected 'best_iteration_details' in extra_data."
            logger.error(error_msg)
            self.session_state["phase_status"]["running experiments"] = False
            return RunResponse(
                content=error_msg,
                event=RunEvent.run_error,
                workflow_id=self.workflow_id,
            )

        best_details = final_exp_workflow_response.extra_data["best_iteration_details"]

        final_code = best_details.get("code", "No code available in best details.")
        final_score = best_details.get("score", 0.0)
        # The 'execution_result' itself is a dict. We might want to store it whole or extract parts.
        final_execution_result_dict = best_details.get("execution_result", {})
        final_log_stdout = final_execution_result_dict.get("stdout", "No stdout.")
        final_log_stderr = final_execution_result_dict.get("stderr", "No stderr.")
        final_exit_code = final_execution_result_dict.get("exit_code", "N/A")

        # Get paths to already saved files
        code_file_path_str = best_details.get("code_file_path")
        result_file_path_str = best_details.get("result_file_path")
        figures_dir_path_str = best_details.get("figures_dir_path")

        if not code_file_path_str:
            logger.warning(
                "Best code file path not found in experiment workflow results. Code will be stored in session state only."
            )
        if not result_file_path_str:
            logger.warning(
                "Best result file path not found in experiment workflow results."
            )
        if not figures_dir_path_str:
            logger.warning(
                "Best figures directory path not found in experiment workflow results."
            )

        # Store final results into the main workflow's session_state
        self.session_state["final_experiment_code"] = final_code
        self.session_state["final_experiment_score"] = final_score
        # Storing structured execution result and individual log parts
        self.session_state["final_experiment_execution_result"] = (
            final_execution_result_dict
        )
        self.session_state["final_experiment_stdout"] = final_log_stdout
        self.session_state["final_experiment_stderr"] = final_log_stderr
        self.session_state["final_experiment_exit_code"] = final_exit_code

        # Store paths to files saved by the sub-workflow
        self.session_state["run_experiments_code_file"] = code_file_path_str
        self.session_state["run_experiments_result_file"] = (
            result_file_path_str  # Changed from log_file to result_file
        )
        self.session_state["run_experiments_figures_dir"] = (
            figures_dir_path_str  # Path to the directory containing figures
        )

        # No longer have 'best_codes_library' in the same way
        self.session_state["best_experiment_details"] = best_details

        logger.info(
            "Running experiments phase completed successfully with ExperimentCodingWorkflow."
        )
        self.session_state["phase_status"]["running experiments"] = True

        final_result_summary = (
            f"Running experiments phase (using ExperimentCodingWorkflow) finished.\n"
            f"Best Iteration Score: {final_score}\n"
            f"Best Code saved to: {code_file_path_str or 'N/A (not provided by sub-workflow)'}\n"
            f"Execution Result (JSON) saved to: {result_file_path_str or 'N/A'}\n"
            f"Figures for best iteration saved in directory: {figures_dir_path_str or 'N/A'}\n"
            f"Exit Code of best run: {final_exit_code}\n"
            f"Stdout (snippet):\n{final_log_stdout}\n"
        )

        return RunResponse(
            content=final_result_summary,
            event=RunEvent.run_completed,
            workflow_id=self.workflow_id,
            extra_data=final_exp_workflow_response.extra_data,
        )

    def _result_interpretation(self, topic) -> Optional[RunResponse]:
        logger.info("Starting result interpretation phase")
        max_tries = self.max_steps  # 使用 Workflow 的 max_steps 作为尝试次数限制
        last_agent_response_in_phase = None  # 用于存储该阶段的最后一个 Agent 响应
        # 获取阶段所需的上下文信息
        phase = "results interpretation"
        lit_review_sum = self.session_state.get(
            "lit_review_sum", "No literature review summary available."
        )
        plan = self.session_state.get("plan", "")
        data_preparation_result = self.session_state.get(
            "data_preparation_result", "No dataset code available."
        )
        research_topic = self.session_state.get("research_topic", "a research topic")
        # 在这里，你需要从之前阶段的 session_state 中获取实验结果
        # 假设实验结果存储在 self.session_state['experimental_results'] 中
        experimental_results = f"""experiment_code:{self.session_state.get('final_experiment_code', '')}
experiment_log:{self.session_state.get('final_experiment_log', '')}
experiment_score:{self.session_state.get('final_experiment_score', '')}
"""
        experimental_results_prompt_section = (
            f"Experimental Results to Interpret:\n{experimental_results}\n\n"
        )
        dialogue_history = self.session_state.get("res_dialogue_history", [])
        dialogue_history_str = "\n".join(dialogue_history)  # 将历史对话转为字符串
        # 轮流对话
        for step in range(max_tries):
            logger.info(f"Results interpretation step {step + 1}/{max_tries}")

            # 1. Postdoc Agent 回应 PhD Student 的对话 (或提供初始指导/问题)
            postdoc_prompt = f"""Current Literature Review: {lit_review_sum}\nPlan:{plan}
experimental_results:{experimental_results_prompt_section}
Task instructions: {self.postdoc.phase_prompt(phase)}\n{self.postdoc.command_descriptions(phase)}
[Objective] Your goal is to perform research on the following topic:{topic}
Dialogue History:{dialogue_history_str}.
Make sure your new output is very different.\nPlease produce a single command below:\n"""
            try:
                logger.info("Calling Postdoc Agent for interpretation...")
                # 调用 Postdoc Agent 的同步运行方法
                postdoc_agent_response = self.postdoc.run(
                    postdoc_prompt,
                    stream=False,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                last_agent_response_in_phase = postdoc_agent_response  # 更新该阶段的最后一个 Agent 响应 (Postdoc 的)

                # 处理 Postdoc Agent 的响应
                if (
                    postdoc_agent_response
                    and postdoc_agent_response.event == RunEvent.run_error
                ):
                    logger.error(
                        f"Postdoc Agent run failed in step {step + 1}: {postdoc_agent_response.content}"
                    )
                    # 提供反馈给 PhD (通过 PhD 的下一轮 Prompt)
                    # Agent 在下一轮会通过其 memory 看到 Postdoc 的运行失败和 Prompt
                    pass  # 继续到 PhD 的调用

                elif (
                    postdoc_agent_response
                    and postdoc_agent_response.event == RunEvent.run_cancelled
                ):
                    logger.warning(
                        f"Postdoc Agent run was cancelled in step {step + 1}"
                    )
                    pass  # 继续到 PhD 的调用

                else:
                    # Postdoc 运行成功，处理其文本内容（可能包含命令）
                    postdoc_resp_text = (
                        postdoc_agent_response.content
                        if isinstance(postdoc_agent_response.content, str)
                        else str(postdoc_agent_response.content)
                    )

                    # 检查 Postdoc 的响应是 DIALOGUE 还是 INTERPRETATION
                    if "```DIALOGUE" in postdoc_resp_text:
                        postdoc_dialogue = extract_prompt(
                            postdoc_resp_text, "DIALOGUE"
                        ).strip()
                        # 如果是 DIALOGUE，Agent 已经通过其 memory 记录了这次对话
                        logger.info(
                            "Postdoc Agent used DIALOGUE command. Continuing conversation."
                        )
                        logger.info(f"Postdoc says: {postdoc_dialogue}")
                        dialogue_history.append(f"Postdoc: {postdoc_dialogue}")
                        dialogue_history_str = "\n".join(
                            dialogue_history
                        )  # 更新历史字符串
                        # 继续到 PhD 的调用 (Agent 内部 memory 更新)
                        pass

                    elif "```INTERPRETATION" in postdoc_resp_text:
                        interpretation_content = extract_prompt(
                            postdoc_resp_text, "INTERPRETATION"
                        ).strip()
                        logger.info("Postdoc Agent submitted final INTERPRETATION.")
                        # 存储最终解释内容到 session_state
                        self.session_state["final_interpretation_content"] = (
                            interpretation_content
                        )
                        # 标记阶段完成
                        self.session_state["phase_status"][
                            "results interpretation"
                        ] = True
                        self.reset_agents()
                        # 阶段完成，返回最后一个 Agent 响应 (这里是 Postdoc 的响应)
                        return last_agent_response_in_phase

                    else:
                        logger.warning(
                            "Postdoc Agent did not use ```DIALOGUE``` or ```INTERPRETATION``` command."
                        )
                        # Postdoc 未使用已知命令，继续到 PhD 的调用
                        pass

            except Exception as e:
                logger.error(f"Error during Postdoc Agent run in step {step + 1}: {e}")
                # 如果 Agent 本身调用失败 (例如 API 错误)
                pass  # 继续到 PhD 的调用

            # 2. PhD Agent 回应 Postdoc 的对话
            phd_prompt = f"""Current Literature Review: {lit_review_sum}\nPlan:{plan}
experimental_results:{experimental_results_prompt_section}
Task instructions: {self.phd.phase_prompt(phase)}\n{self.phd.command_descriptions(phase)}
[Objective] Your goal is to perform research on the following topic:{topic}
Dialogue History:{dialogue_history_str}.
Make sure your new output is very different.\nPlease produce a single command below:\n"""
            try:
                logger.info("Calling PhD Student Agent...")
                phd_agent_response = self.phd.run(
                    phd_prompt,
                    stream=False,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                # 更新该阶段的最后一个 Agent 响应
                last_agent_response_in_phase = phd_agent_response

                # 处理 PhD Agent 的响应
                if (
                    phd_agent_response
                    and phd_agent_response.event == RunEvent.run_error
                ):
                    logger.error(
                        f"PhD Agent run failed in step {step + 1}: {phd_agent_response.content}"
                    )
                    pass  # 继续循环

                elif (
                    phd_agent_response
                    and phd_agent_response.event == RunEvent.run_cancelled
                ):
                    logger.warning(f"PhD Agent run was cancelled in step {step + 1}")
                    pass  # 继续循环

                else:
                    # PhD 运行成功，处理其文本内容（可能包含命令）
                    phd_resp_text = (
                        phd_agent_response.content
                        if isinstance(phd_agent_response.content, str)
                        else str(phd_agent_response.content)
                    )

                    # 检查 PhD 的响应是 DIALOGUE
                    if "```DIALOGUE" in phd_resp_text:
                        # 如果是 DIALOGUE，Agent 已经通过其 memory 记录了这次对话
                        logger.info(
                            "PhD Student used DIALOGUE command. Continuing conversation."
                        )

                        phd_dialogue = extract_prompt(phd_resp_text, "DIALOGUE").strip()
                        # 如果是 DIALOGUE，Agent 已经通过其 memory 记录了这次对话
                        logger.info(
                            "Postdoc Agent used DIALOGUE command. Continuing conversation."
                        )
                        logger.info(f"PhD says: {phd_dialogue}")
                        dialogue_history.append(f"Postdoc: {phd_dialogue}")
                        pass  # 继续循环，进入下一轮对话 (Postdoc 先发言)

                    else:
                        logger.warning("PhD Agent did not use ```DIALOGUE``` command.")
                        # PhD 未使用已知命令，继续循环
                        pass

            except Exception as e:
                logger.error(f"Error during PhD Agent run in step {step + 1}: {e}")
                pass  # 继续循环

        # 如果循环因为达到最大步数而结束，并且 INTERPRETATION 命令未被使用
        logger.warning(
            f"Results interpretation phase completed (Max tries {max_tries} reached) without a final INTERPRETATION command."
        )
        prompt = f"""Current Literature Review: {lit_review_sum}\nPlan:{plan}
experimental_results:{experimental_results_prompt_section}
Task instructions: {self.postdoc.phase_prompt(phase)}\n{self.postdoc.command_descriptions(phase)}
[Objective] Your goal is to perform research on the following topic:{topic}
Dialogue History:{dialogue_history_str}.
It's time for the final move, and now you have to use ```INTERPRETATION``` command.\nPlease produce a single command below:\n"""
        postdoc_agent_response = self.postdoc.run(
            prompt, stream=False, user_id=self.user_id, session_id=self.session_id
        )
        last_agent_response_in_phase = postdoc_agent_response
        postdoc_resp_text = (
            postdoc_agent_response.content
            if isinstance(postdoc_agent_response.content, str)
            else str(postdoc_agent_response.content)
        )
        interpretation_content = extract_prompt(
            postdoc_resp_text, "INTERPRETATION"
        ).strip()
        # 标记阶段完成 (即使没有最终解释)
        self.session_state["phase_status"]["results interpretation"] = True
        self.session_state["final_interpretation_content"] = interpretation_content

        # 返回该阶段的最后一个 Agent 响应
        return last_agent_response_in_phase  # 返回该阶段的最后一个 Agent 响应 (可能是 PhD 或 Postdoc 的，取决于最后是哪一方说话并成功返回，或 None)

    def _paper_writing(self, topic) -> Optional[RunResponse]:
        workflow = PaperWritingWorkflow(
            base_output_path=self.lab_dir,
            max_iterations=self.max_steps,  # Reduced iterations for a quicker demo
            min_score_to_accept=8.5,
            debug_mode=True,  # Enable debug output for the workflow
        )
        lit_review_summary = self.session_state["lit_review_sum"]
        lit_review_paper = self.session_state["lit_review"]
        exp_code = self.session_state["final_experiment_code"]
        exp_result = self.session_state["final_experiment_log"]
        res_interpretation = self.session_state["final_interpretation_content"]
        plan = self.session_state["plan"]

        results_iterator = workflow.run(
            lit_review_sum=lit_review_summary,
            lit_review_paper=lit_review_paper,
            plan=plan,
            exp_code=exp_code,
            exp_result=exp_result,
            res_interpretation=res_interpretation,
            experiment_figures_source_dir=str(self.lab_dir + "\\" + "figures"),
            stream_intermediate_steps=True,  # Set to True to see progress in real-time
        )

        final_response = None
        for chunk in results_iterator:
            logger.debug(chunk.content)
            if chunk.event in [RunEvent.run_completed, RunEvent.run_error]:
                final_response = chunk
                logger.info("--- Workflow Finished ---")

        if final_response:
            logger.info(f"Final Status: {final_response.event}")
            logger.info(f"Final Message: {final_response.content}")
            if final_response.images:
                logger.info(
                    f"Generated images paths: {[img.url for img in final_response.images]}"
                )
        pass

    def _generate_lit_review_summary(self) -> str:
        """生成文献综述总结 (可以是另一个 Agent 或 Tool 调用)"""
        logger.info("Generating literature review summary...")
        if not self.session_state.get("lit_review"):
            logger.warning("No papers in literature review to summarize.")
            return "No papers were found for the literature review."

        summaries = "\n".join(
            [
                f"- {p.get('arxiv_id', 'N/A')}: {p.get('summary', 'No summary provided.')}"
                for p in self.session_state["lit_review"]
            ]
        )
        prompt = f"Based on the following papers added to the literature review for the topic '{self.session_state['research_topic']}', write a concise summary of the current state of research, key findings, and identify potential gaps or areas for future work:\n\n{summaries}\n\nSummary:"
        system_prompt = (
            "You are an expert research summarizer. Provide a well-structured summary."
        )

        try:
            lit_review_summary = query_model(
                system_prompt=system_prompt,
                prompt=prompt,
            )
            logger.info("Literature review summary generated successfully.")
            return lit_review_summary
        except Exception as e:
            logger.error(f"Failed to generate literature review summary: {e}")
            return f"Failed to generate summary due to an error: {e}"

    def reset_agents(self):
        self.phd.new_session()
        self.postdoc.new_session()
        self.swe.new_session()
        self.mle.new_session()
