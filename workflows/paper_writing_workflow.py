import re  # for parsing PDF path
import os
import shutil
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Tuple

import json
from agno.agent import Agent
from agno.workflow import Workflow, RunResponse
from agno.run.response import RunEvent
from agno.utils.log import logger
from agno.tools.function import Function  # For compile_tool type hint

# 从新的工具位置导入
from tools.latex_compiler import (
    compile_latex,
    generate_graph_from_dot,
    generate_references_bib,
    propose_figures_and_content,
    generate_dot_from_description,
)
from agents.paper_agents import (
    get_paper_agent,
    LatexOutput,
    get_reward_agent,
    PaperScore,
)
from config import OPENAI_MODEL_NAME  # Add this import


class PaperWritingWorkflow(Workflow):
    description: str = (
        "A workflow to iteratively write a LaTeX paper using two agents and compilation/graph generation tools."
    )

    paper_agent_initial: Agent
    paper_agent_revise: Agent
    reward_agent: Agent
    compile_tool: Function
    generate_graph_tool: Function
    generate_bib_tool: Function
    propose_figures_tool: Function
    generate_dot_tool: Function

    def __init__(
        self,
        max_iterations: int = 3,
        min_score_to_accept: float = 8.0,
        # base_output_path is the main work directory (e.g., C:\project\AgentLab_demo)
        base_output_path: str = os.getcwd(),
        debug_mode: bool = False,
        paper_agent_model_id: str = OPENAI_MODEL_NAME,  # Updated default
        reward_agent_model_id: str = OPENAI_MODEL_NAME,  # Updated default
        **kwargs,
    ):
        super().__init__(
            name="PaperWritingWorkflow",
            description="Generates a LaTeX paper iteratively.",
            debug_mode=debug_mode,
            **kwargs,
        )

        self.max_iterations: int = max_iterations
        self.min_score_to_accept: float = min_score_to_accept
        self.base_output_dir: Path = Path(base_output_path)

        # Ensure the base output directory for all artifacts exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.paper_agent_initial = get_paper_agent(
            model_id=paper_agent_model_id, iteration=0
        )
        self.paper_agent_revise = get_paper_agent(
            model_id=paper_agent_model_id, iteration=1
        )
        self.reward_agent = get_reward_agent(model_id=reward_agent_model_id)

        # Initialize tools (assuming they are decorated with @tool and are now Function objects)
        self.compile_tool = compile_latex
        self.generate_graph_tool = generate_graph_from_dot
        self.generate_bib_tool = generate_references_bib
        self.propose_figures_tool = propose_figures_and_content
        self.generate_dot_tool = generate_dot_from_description

    def run(
        self,
        lit_review_sum: str,
        lit_review_paper: str,
        plan: str,
        exp_code: str,
        exp_result: str,
        res_interpretation: str,
        # Source directory for pre-existing experiment figures (e.g., from exp_result)
        experiment_figures_source_dir: Optional[str] = None,
        stream_intermediate_steps: bool = False,
    ) -> Iterator[RunResponse]:

        best_latex_code: Optional[str] = None
        best_score: float = -1.0
        best_compilation_status: Optional[str] = (
            "Compilation not attempted for any version yet."
        )
        current_feedback: Optional[str] = "No feedback yet (first iteration)."
        compilation_status: Optional[str] = "Not compiled yet."
        current_latex_code: Optional[str] = None
        # final_pdf_path_str: str = "N/A" # This will be determined at the very end
        references_bib_content: str = ""
        generated_figure_paths: List[str] = (
            []
        )  # To store paths of DOT generated figures
        generated_diagram_details_for_prompt: List[str] = (
            []
        )  # To store details for the agent prompt
        experimental_figure_details_for_prompt: List[str] = (
            []
        )  # For figures from exp_result

        # --- 设置最终输出目录 ---
        # 确保这个目录存在，所有最终输出将在这里
        final_paper_output_root_dir = self.base_output_dir / "paper_dir"
        final_paper_output_root_dir.mkdir(parents=True, exist_ok=True)
        # 清理之前可能存在的同名最终输出文件
        for item in final_paper_output_root_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        # Define the common figures directory within the base_output_dir for all figures
        common_figures_dir = self.base_output_dir / "figures"
        common_figures_dir.mkdir(parents=True, exist_ok=True)

        # List to keep track of temporary iteration-specific build directories
        temp_iteration_build_dirs: List[Path] = []

        # --- 0.1 Generate references.bib ---
        yield RunResponse(
            event=RunEvent.run_response, content="📚 Generating references.bib...\n"
        )
        references_bib_content = self.generate_bib_tool.entrypoint(
            lit_review_paper_content=lit_review_paper, lit_review_sum=lit_review_sum
        )
        bib_file_path = self.base_output_dir / "references.bib"
        try:
            bib_file_path.write_text(references_bib_content, encoding="utf-8")
            yield RunResponse(
                event=RunEvent.run_response,
                content=f"✅ references.bib generated and saved to '{bib_file_path}'.\n",
            )
        except Exception as e:
            yield RunResponse(
                event=RunEvent.run_error,
                content=f"❌ Error saving references.bib: {e}\n",
            )
            return  # Critical error, cannot proceed

        # --- 0.2 Propose and Generate Diagrams (DOT code) ---
        initial_paper_structure_for_diagram_proposal = f"""
        Abstract: [To be generated based on overall paper content]
        Introduction: [To be generated, outlining motivation, problem, contributions, and paper structure]
        Related Work: [To be generated from lit_review_sum and lit_review_paper, comparing and contrasting with existing work]
        Methods: [To be generated from exp_code and plan, detailing the proposed methods, algorithms, and system architecture. Conceptual diagrams often go here.]
        Experiments: [This comprehensive section will cover: Experimental Setup (datasets, evaluation metrics, baselines, implementation specifics - drawn from exp_code and plan); Presentation of Results (figures, tables from exp_result); and DETAILED ANALYSIS AND INTERPRETATION of experimental outcomes, primarily from res_interpretation.]
        Conclusion: [High-level summary of the entire paper - problem, methods, key findings/contributions - and future work. Based on overall paper content.]
        References: [To be generated from references.bib]

        Note: The 'Research Plan: {plan}' provided below is for additional context on the research workflow and content, not to define the paper's section structure itself.
        """
        yield RunResponse(
            event=RunEvent.run_response,
            content="🎨 Proposing conceptual diagrams based on paper structure and content...\n",
        )
        proposed_figures_json_str = self.propose_figures_tool.entrypoint(
            paper_current_draft=initial_paper_structure_for_diagram_proposal,
            research_plan=plan,
        )
        try:
            proposed_figures: List[Dict[str, str]] = json.loads(
                proposed_figures_json_str
            )
            for fig_num, figure_proposal in enumerate(proposed_figures):
                fig_title = figure_proposal.get("title", f"Diagram {fig_num+1}")
                fig_description = figure_proposal.get(
                    "description", "No description provided."
                )
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"➡️ Proposed Figure: '{fig_title}'. Description: {fig_description}\n",
                )

                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"✍️ Generating DOT code for '{fig_title}'...\n",
                )
                dot_code = self.generate_dot_tool.entrypoint(
                    figure_title=fig_title, figure_description=fig_description
                )
                if not dot_code or not dot_code.strip():
                    yield RunResponse(
                        event=RunEvent.run_error,
                        content=f"❌ No DOT code generated for '{fig_title}'. Skipping image generation.\n",
                    )
                    continue

                image_filename = f"{fig_title.replace(' ', '_').lower().replace('.', '_').replace(':', '_').replace('/', '_')}.png"  # Clean filename
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"🖼️ Drawing '{fig_title}' as '{image_filename}'...\n",
                )
                graph_image_path = self.generate_graph_tool.entrypoint(
                    directory=str(common_figures_dir),  # Save generated graphs here
                    dot_code=dot_code,
                    image_filename=image_filename,
                    format="png",
                )
                if graph_image_path:
                    generated_figure_paths.append(graph_image_path)
                    diagram_detail = f'- Title: "{fig_title}", Filename: "{image_filename}", Description: "{fig_description}"'
                    generated_diagram_details_for_prompt.append(diagram_detail)
                    yield RunResponse(
                        event=RunEvent.run_response,
                        content=f"✅ Diagram '{fig_title}' saved to '{graph_image_path}'.\n",
                    )
                else:
                    yield RunResponse(
                        event=RunEvent.run_error,
                        content=f"❌ Failed to generate image for '{fig_title}'. DOT code:\n```dot\n{dot_code}\n```\n",
                    )

        except json.JSONDecodeError:
            yield RunResponse(
                event=RunEvent.run_error,
                content=f"❌ LLM returned invalid JSON for figure proposals: {proposed_figures_json_str}\n",
            )
        except Exception as e:
            yield RunResponse(
                event=RunEvent.run_error,
                content=f"❌ Error during diagram generation: {e}\n",
            )

        # --- 0.3 Copy experiment figures from exp_result and prepare for prompt ---
        yield RunResponse(
            event=RunEvent.run_response,
            content="📊 Processing experimental figures from results...\n",
        )
        # Assuming exp_result contains lines like "Figure saved to: /absolute/path/to/figure.png"
        # Or from the example: "Figure saved to: C:\\project\\ScienceAgent\\paper_dir\\figures\\Accuracy_vs_Complexity_Adult.pdf"
        # We need to handle both .png and .pdf if they are used.
        # Regex to find figure paths. Handles spaces in paths and various extensions.
        # Looks for a line starting with something like "Figure saved to: " or just a path ending with a common image extension.
        figure_path_pattern = re.compile(
            r"(?:Figure saved to:|Plots? saved to:|Plot is saved to:|Figure available at:)\s*"  # Prefix
            r"(?P<path>"  # Start path group
            r"[\"']?"  # Optional opening quote
            r"([^\"']+\.(?:png|pdf|jpg|jpeg|svg))"  # The actual path and extension
            r"[\"']?"  # Optional closing quote
            r")",  # End path group
            re.IGNORECASE,
        )

        # Also, sometimes the path might be directly printed if the script uses a library that logs it.
        direct_path_pattern = re.compile(
            r"(?P<path>"
            r"(?:[a-zA-Z]:)?"  # Optional drive letter for Windows
            r"[^\"'\s<>]+\.(?:png|pdf|jpg|jpeg|svg)"  # Path characters and extension
            r")"
        )

        found_exp_figure_paths: List[str] = []
        if exp_result:
            for line in exp_result.splitlines():
                match = figure_path_pattern.search(line)
                if match:
                    found_exp_figure_paths.append(match.group("path").strip("'\""))
                else:
                    # Try to find direct paths if the specific prefix is not there
                    direct_matches = direct_path_pattern.findall(line)
                    for direct_match_path in direct_matches:
                        # Basic check to avoid matching URLs or very short non-paths
                        if Path(direct_match_path).is_file() and Path(
                            direct_match_path
                        ).name not in [Path(p).name for p in found_exp_figure_paths]:
                            found_exp_figure_paths.append(
                                direct_match_path.strip("'\"")
                            )

        if (
            experiment_figures_source_dir
            and Path(experiment_figures_source_dir).is_dir()
        ):
            yield RunResponse(
                event=RunEvent.run_response,
                content=f"Scanning provided experiment_figures_source_dir: {experiment_figures_source_dir}\n",
            )
            for item in Path(experiment_figures_source_dir).iterdir():
                if item.is_file() and item.suffix.lower() in [
                    ".png",
                    ".pdf",
                    ".jpg",
                    ".jpeg",
                    ".svg",
                ]:
                    if str(item.resolve()) not in found_exp_figure_paths:
                        found_exp_figure_paths.append(str(item.resolve()))

        # Deduplicate paths found
        unique_exp_figure_paths = sorted(list(set(found_exp_figure_paths)))

        for i, original_fig_path_str in enumerate(unique_exp_figure_paths):
            original_fig_path = Path(original_fig_path_str)
            # Ensure the path is absolute for exists() and is_file() checks
            if not original_fig_path.is_absolute():
                # This case needs careful handling. If paths in exp_result are relative,
                # they might be relative to the script that produced them, or the project root.
                # For now, we'll assume if it's not absolute, it might be relative to common_figures_dir
                # or that experiment_figures_source_dir covers it.
                # A more robust solution might require making them absolute based on a known root.
                logger.info(
                    f"Found relative experimental figure path: {original_fig_path_str}. Assuming it's accessible or will be resolved."
                )
                # We'll proceed assuming the agent can find it via `../figures/{original_fig_path.name}`
                # if it's already in the common_figures_dir.

            if original_fig_path.exists() and original_fig_path.is_file():
                # Figure exists. We assume it's already in the correct common_figures_dir
                # or accessible to the LaTeX compilation process via `../figures/`
                fig_filename_for_latex = (
                    original_fig_path.name
                )  # Use the original filename

                # Check if the figure is actually within the common_figures_dir.
                # This is a sanity check; if not, the LaTeX might fail unless paths are handled perfectly.
                try:
                    resolved_common_figures_dir = common_figures_dir.resolve()
                    resolved_original_fig_path_parent = (
                        original_fig_path.parent.resolve()
                    )
                    if resolved_original_fig_path_parent != resolved_common_figures_dir:
                        logger.warning(
                            f"Experimental figure '{original_fig_path_str}' is not directly in common_figures_dir ('{str(common_figures_dir)}'). "
                            f"Its parent is '{str(original_fig_path.parent)}'. Ensure LaTeX can access it via '../figures/{fig_filename_for_latex}'."
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not fully resolve paths for figure '{original_fig_path_str}' check against common_figures_dir: {e}"
                    )

                detail = f'- Filename in paper: "{fig_filename_for_latex}", Type: Experimental Result (Original path: "{original_fig_path_str}")'
                experimental_figure_details_for_prompt.append(detail)
                if stream_intermediate_steps:
                    yield RunResponse(
                        event=RunEvent.run_response,
                        content=f"✅ Identified experimental figure: '{fig_filename_for_latex}' (from path: {original_fig_path_str}). Will be used directly.\n",
                    )
            else:
                # If the absolute path doesn't exist, try resolving it relative to common_figures_dir as a fallback for exp_result entries that are just filenames
                potential_path_in_common = common_figures_dir / original_fig_path.name
                if (
                    potential_path_in_common.exists()
                    and potential_path_in_common.is_file()
                ):
                    fig_filename_for_latex = original_fig_path.name
                    detail = f'- Filename in paper: "{fig_filename_for_latex}", Type: Experimental Result (Found in common_figures_dir)'
                    experimental_figure_details_for_prompt.append(detail)
                    if stream_intermediate_steps:
                        yield RunResponse(
                            event=RunEvent.run_response,
                            content=f"✅ Identified experimental figure in common_figures_dir: '{fig_filename_for_latex}'. Will be used directly.\n",
                        )
                else:
                    logger.info(
                        f"Experimental figure path/name '{original_fig_path_str}' does not exist or is not a file, nor found as '{original_fig_path.name}' in common_figures_dir."
                    )
                    if stream_intermediate_steps:
                        yield RunResponse(
                            event=RunEvent.run_response,
                            content=f"⚠️ Experimental figure path/name '{original_fig_path_str}' not found or not a file. It will be omitted.\n",
                        )
                continue

        # --- Prepare initial prompt for paper agent with all generated context ---
        generated_diagrams_section = "No conceptual diagrams were specifically generated from descriptions in this run."
        if generated_diagram_details_for_prompt:
            generated_diagrams_section = (
                "The following conceptual diagrams were generated from descriptions and are available in the `../figures/` directory. "
                "You should try to incorporate them meaningfully into the paper, referencing them by their filenames (usually in Methodology or related sections):\n"
                + "\n".join(generated_diagram_details_for_prompt)
            )

        experimental_figures_section = "No specific experimental result figures were identified from the experiment output."
        if experimental_figure_details_for_prompt:
            experimental_figures_section = (
                "The following experimental result figures were identified from the experiment output and are available in the `../figures/` directory. "
                "You should try to incorporate them meaningfully into the paper, referencing them by their filenames (usually in Experiments or Results sections):\n"
                + "\n".join(experimental_figure_details_for_prompt)
            )

        initial_paper_agent_prompt_data = f"""
Input Materials for Paper Generation:

**Conceptual Diagrams (e.g., methodology, architecture - generated from descriptions):**
{generated_diagrams_section}

**Experimental Result Figures (generated by experiment code):**
{experimental_figures_section}

1. Literature Review Summary (`lit_review_sum`):
{lit_review_sum}

2. Cited Papers Details (`lit_review_paper` - titles, authors, abstracts):
{lit_review_paper}

3. Research Plan and Paper Structure (`plan`):
{plan}

4. Experiment Code (`exp_code` - for context, focus on methodology description):
{exp_code}

5. Experiment Results (`exp_result` - raw data or summaries):
{exp_result}

6. Result Interpretation (`res_interpretation` - explanation of findings):
{res_interpretation}

7. Figures Available Path (All figures, including experiment results and generated diagrams, are in this directory, relative path for LaTeX is '../figures/'):
{str(common_figures_dir.relative_to(self.base_output_dir))}/ (for LaTeX: ../figures/)
   - Conceptual diagram filenames are listed above.
   - Experimental result figure filenames are listed above.

8. References BibTeX Content (`references_bib_content`):
```bibtex
{references_bib_content}
```
"""

        # --- Iterative Paper Writing Loop ---
        for iteration_count in range(self.max_iterations):
            iteration_message = (
                f"--- Iteration {iteration_count + 1}/{self.max_iterations} ---"
            )
            logger.info(iteration_message)
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"\n\n**{iteration_message}**\n",
                )

            # Create a temporary subdirectory for this iteration's compilation outputs
            # This directory will be unique to this iteration and will be cleaned up later
            temp_iter_compile_subdir_name = f"temp_compile_iter_{iteration_count + 1}"
            temp_iter_compile_dir = self.base_output_dir / temp_iter_compile_subdir_name
            temp_iter_compile_dir.mkdir(parents=True, exist_ok=True)
            temp_iteration_build_dirs.append(
                temp_iter_compile_dir
            )  # Add to list for cleanup

            # 1. Paper Agent generates/revises paper
            agent_to_use: Agent
            paper_agent_prompt_content: str
            if iteration_count == 0:
                agent_to_use = self.paper_agent_initial
                paper_agent_prompt_content = (
                    f"Task: Generate First Draft.\n\n{initial_paper_agent_prompt_data}"
                )
            else:
                agent_to_use = self.paper_agent_revise
                paper_agent_prompt_content = f"""Task: Revise Paper (Iteration {iteration_count+1}).
Original Requirements (for context):
{initial_paper_agent_prompt_data}

Previous LaTeX Code (Iteration {iteration_count}):
```latex
{current_latex_code if current_latex_code else "No previous LaTeX code available."}
```

Compilation Result of Previous LaTeX Code:
{compilation_status if compilation_status else "Previous code not compiled."}

Feedback from Evaluator (for Iteration {iteration_count}, leading to this revision):
{current_feedback if current_feedback else "No previous feedback."}
"""
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"🖋️ Paper Agent '{agent_to_use.name}' is generating LaTeX (Iteration {iteration_count + 1})...\n",
                )

            paper_agent_response: RunResponse = agent_to_use.run(
                paper_agent_prompt_content, stream=False
            )

            if not isinstance(paper_agent_response.content, LatexOutput):
                error_msg = f"Paper Agent (iteration {iteration_count+1}) failed to return LatexOutput. Received: {type(paper_agent_response.content)}. Content: {str(paper_agent_response.content)[:500]}..."
                logger.error(error_msg)
                if stream_intermediate_steps:
                    yield RunResponse(event=RunEvent.run_error, content=error_msg)
                # If agent returned raw string that looks like LaTeX, try to use it
                if (
                    isinstance(paper_agent_response.content, str)
                    and "\\documentclass" in paper_agent_response.content
                ):
                    current_latex_code = paper_agent_response.content
                    generation_explanation = "Agent returned raw LaTeX string instead of LatexOutput object. Attempting to proceed."
                    logger.warning(generation_explanation)
                else:  # Otherwise, it's a critical failure in output format
                    current_feedback = "Critical error: Paper agent did not produce valid LaTeX output structure. Please try again, ensuring the output is a JSON with `latex_code` and `explanation` fields."
                    compilation_status = "Skipped due to invalid LaTeX output structure from Paper Agent."
                    continue
            else:
                current_latex_code = paper_agent_response.content.latex_code
                generation_explanation = paper_agent_response.content.explanation

            if (
                not current_latex_code
                or not isinstance(current_latex_code, str)
                or not current_latex_code.strip()
            ):
                error_msg = f"Paper Agent (iteration {iteration_count+1}) returned empty LaTeX code."
                logger.error(error_msg)
                if stream_intermediate_steps:
                    yield RunResponse(event=RunEvent.run_error, content=error_msg)
                current_feedback = "The paper agent returned empty LaTeX code. Please generate the complete paper content."
                compilation_status = "Skipped due to empty LaTeX code from Paper Agent."
                continue

            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"📄 Paper Agent (iteration {iteration_count + 1}) generated LaTeX. Explanation: {generation_explanation or 'N/A'}\n",
                )

            # 2. Compile LaTeX (into the temporary directory for this iteration)
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"⚙️ Compiling LaTeX (iteration {iteration_count + 1}) into '{temp_iter_compile_dir}'...\n",
                )

            compilation_status = self.compile_tool.entrypoint(
                latex_code=current_latex_code,
                base_output_path=str(
                    temp_iter_compile_dir
                ),  # 将临时目录作为 base_output_path 传入
            )

            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"🛠️ Compilation result (iteration {iteration_count + 1}):\n```text\n{compilation_status}\n```\n",
                )

            # 3. Reward Agent scores the paper
            reward_agent_prompt = f"""
Original Research Plan:
{plan}

Generated LaTeX Code (Current Iteration {iteration_count+1}):
```latex
{current_latex_code}
```

Compilation Result of Current LaTeX Code:
{compilation_status}

Please evaluate this version.
"""
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"⚖️ Reward Agent (iteration {iteration_count + 1}) is scoring...\n",
                )

            reward_agent_response: RunResponse = self.reward_agent.run(
                reward_agent_prompt, stream=False
            )

            if not isinstance(reward_agent_response.content, PaperScore):
                error_msg = f"Reward Agent (iteration {iteration_count+1}) did not return PaperScore. Received: {type(reward_agent_response.content)}. Content: {str(reward_agent_response.content)[:200]}..."
                logger.error(error_msg)
                if stream_intermediate_steps:
                    yield RunResponse(event=RunEvent.run_error, content=error_msg)
                current_feedback = "Critical error: Reward agent did not produce valid scoring output. Please try again, ensuring the output is a JSON with `score` and `feedback` fields."
                compilation_status = "Skipped due to invalid Reward Agent output."  # Ensure status is updated
                continue

            current_score = reward_agent_response.content.score
            current_feedback = reward_agent_response.content.feedback
            compilation_fixed_by_suggestion = (
                reward_agent_response.content.compilation_fixed
            )
            positive_aspects = reward_agent_response.content.positive_aspects
            key_issues_to_address = reward_agent_response.content.key_issues_to_address

            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=(
                        f"📊 Reward Agent (iteration {iteration_count + 1}) score: {current_score}/10.\n"
                        f"Feedback: {current_feedback}\n"
                        f"Compilation fixed by suggestion: {compilation_fixed_by_suggestion}\n"
                        f"Positive Aspects: {positive_aspects or 'N/A'}\n"
                        f"Key Issues to Address: {', '.join(key_issues_to_address) if key_issues_to_address else 'N/A'}\n"
                    ),
                )

            # Update best score and code
            compilation_successful_now = "Compilation successful" in compilation_status

            # Logic to keep track of the BEST LaTeX code
            # Prioritize compilable versions. If both compilable, pick higher score.
            # If no compilable yet, pick highest score from non-compilable.
            if compilation_successful_now:
                if (
                    best_latex_code is None
                    or current_score >= best_score
                    or (
                        current_score == best_score
                        and "Compilation successful"
                        not in (best_compilation_status or "")
                    )
                ):
                    best_score = current_score
                    best_latex_code = current_latex_code
                    best_compilation_status = compilation_status
                    logger.info(
                        f"New best version (compilable). Score: {best_score} at iteration {iteration_count + 1}"
                    )
                    if stream_intermediate_steps:
                        yield RunResponse(
                            event=RunEvent.run_response,
                            content=f"🏆 New best (compilable) version found! Score: {best_score}/10\n",
                        )
            elif best_latex_code is None or current_score >= best_score:
                # If no compilable version found yet, take the highest scored non-compilable one
                best_score = current_score
                best_latex_code = current_latex_code
                best_compilation_status = compilation_status
                logger.info(
                    f"New best score (non-compilable): {best_score} at iteration {iteration_count + 1}"
                )
                if stream_intermediate_steps:
                    yield RunResponse(
                        event=RunEvent.run_response,
                        content=f"🏆 New best version (compilation failed). Score: {best_score}/10\n",
                    )

            # Check stopping condition
            if current_score >= self.min_score_to_accept and compilation_successful_now:
                logger.info(
                    f"Score {current_score} meets acceptance threshold ({self.min_score_to_accept}) and compiled successfully. Stopping."
                )
                if stream_intermediate_steps:
                    yield RunResponse(
                        event=RunEvent.run_response,
                        content=f"✅ Score ({current_score}/10) meets acceptance threshold and compiled successfully. Finalizing.\n",
                    )
                break

            if iteration_count == self.max_iterations - 1:
                logger.info("Max iterations reached.")
                if stream_intermediate_steps:
                    yield RunResponse(
                        event=RunEvent.run_response,
                        content="🏁 Max iterations reached. Using best result achieved so far.\n",
                    )

        # --- 5. Final compilation and output of the best code ---
        final_message: str
        if best_latex_code:
            final_compile_dir = final_paper_output_root_dir  # Use the run-specific root dir for final output

            logger.info(
                f"Compiling the best LaTeX code (score: {best_score}) into '{final_compile_dir}'"
            )
            if stream_intermediate_steps:
                yield RunResponse(
                    event=RunEvent.run_response,
                    content=f"⚙️ Finalizing and compiling best paper (score: {best_score}/10) into '{final_compile_dir}'...\n",
                )

            final_compilation_status = self.compile_tool.entrypoint(
                latex_code=best_latex_code,
                base_output_path=str(
                    final_compile_dir
                ),  # 直接将最终输出目录作为 base_output_path
            )

            final_pdf_absolute_path = Path("N/A")
            if "PDF successfully generated: " in final_compilation_status:
                try:
                    match = re.search(
                        r"PDF successfully generated:\s*(.+?)(?:\n|$)",
                        final_compilation_status,
                    )
                    if match:
                        final_pdf_path_str = match.group(1).strip()
                        final_pdf_absolute_path = Path(final_pdf_path_str).resolve()
                    else:  # Fallback if regex fails but success is reported
                        final_pdf_path_str = (
                            final_compilation_status.split(
                                "PDF successfully generated: "
                            )[1]
                            .split("\n")[0]
                            .strip()
                        )
                        final_pdf_absolute_path = Path(final_pdf_path_str).resolve()
                except Exception as e:
                    logger.error(
                        f"Could not parse PDF path from final compilation status: {e}"
                    )
            elif "Compilation successful" in final_compilation_status:
                potential_pdf_path = final_compile_dir / "paper.pdf"
                if potential_pdf_path.exists():
                    final_pdf_absolute_path = potential_pdf_path.resolve()
            else:
                final_pdf_absolute_path = Path("N/A")

            logger.info(f"Final compilation status: {final_compilation_status}")
            final_message = f"Paper writing workflow completed. Best score achieved: {best_score}/10.\n"
            final_message += f"Final chosen LaTeX code compilation status: {final_compilation_status}\n"
            if (
                final_pdf_absolute_path.exists()
                and str(final_pdf_absolute_path) != "N/A"
            ):
                final_message += f"Final PDF available at: {final_pdf_absolute_path}"
            else:
                final_message += "Final PDF was not found at the expected location or compilation failed."

            yield RunResponse(event=RunEvent.run_completed, content=final_message)
        else:
            final_message = "Workflow finished, but no usable LaTeX code was generated after all iterations."
            logger.warning(final_message)
            yield RunResponse(event=RunEvent.run_error, content=final_message)

        # --- 6. Cleanup temporary iteration-specific build directories ---
        yield RunResponse(
            event=RunEvent.run_response,
            content="\n🧹 Cleaning up temporary build directories...\n",
        )
        for temp_dir in temp_iteration_build_dirs:
            try:
                if temp_dir.is_dir():
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up: {temp_dir}")
                else:
                    logger.debug(
                        f"Temporary directory {temp_dir} does not exist or is not a directory."
                    )
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")
        yield RunResponse(
            event=RunEvent.run_response,
            content="✅ Temporary build directories cleaned up.\n",
        )
