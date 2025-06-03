import os
import subprocess
import json
from utils.api_clients import query_model
from utils.text_processing import extract_prompt
from agno.tools import tool
from agno.utils.log import logger


@tool
def compile_latex(latex_code, base_output_path, timeout=60):
    """
    Compiles LaTeX code using latexmk after cleaning up previous compilation files.

    Args:
        latex_code (str): The LaTeX code to compile.
        base_output_path (str): The base directory where compilation will occur.
                                A subdirectory 'latex_build' will be created here.
        timeout (int): Timeout in seconds for the latexmk compilation process.

    Returns:
        str: A message indicating compilation success or failure, including logs.
    """
    build_dir = base_output_path
    tex_file_name = "paper.tex"
    tex_file_path = os.path.join(build_dir, tex_file_name)
    pdf_file_path = os.path.join(build_dir, "paper.pdf")
    os.makedirs(build_dir, exist_ok=True)
    compile_log = ["--- Starting LaTeX Compilation ---"]
    try:
        cleanup_process = subprocess.run(
            ["latexmk", "-C"],
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        compile_log.append("--- Cleanup (latexmk -C) Attempt ---")
        if cleanup_process.stdout:
            compile_log.append(f"Cleanup STDOUT:\n{cleanup_process.stdout}")
        if cleanup_process.stderr:
            compile_log.append(f"Cleanup STDERR:\n{cleanup_process.stderr}")

    except Exception as e:
        compile_log.append(f"Warning: Error during cleanup phase: {str(e)}")

    try:
        with open(tex_file_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
        compile_log.append(f"LaTeX code written to {tex_file_path}")
    except IOError as e:
        return f"[PREPARATION ERROR]: Could not write to {tex_file_path}: {str(e)}"

    try:
        result = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-file-line-error",
                tex_file_name,
            ],
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        compile_log.append("--- latexmk Compilation SUCCESS ---")
        if result.stdout:
            compile_log.append(f"latexmk STDOUT:\n{result.stdout}")
        if result.stderr:
            compile_log.append(f"latexmk STDERR (Warnings):\n{result.stderr}")

        if os.path.exists(pdf_file_path):
            compile_log.append(f"PDF successfully generated: {pdf_file_path}")
        else:
            compile_log.append(
                f"Warning: PDF file not found at {pdf_file_path} despite successful latexmk run."
            )

        return "\n".join(compile_log)

    except subprocess.TimeoutExpired:
        compile_log.append(
            f"[CODE EXECUTION ERROR]: Compilation (latexmk) timed out after {timeout} seconds."
        )
        log_file_path = os.path.join(build_dir, "paper.log")
        if os.path.exists(log_file_path):
            try:
                with open(
                    log_file_path, "r", encoding="utf-8", errors="replace"
                ) as log_f:
                    compile_log.append(
                        f"\n--- Contents of paper.log (on timeout) ---\n{log_f.read()}"
                    )
            except Exception as e_log:
                compile_log.append(f"Could not read paper.log on timeout: {e_log}")
        return "\n".join(compile_log)

    except subprocess.CalledProcessError as e:
        compile_log.append(
            f"[CODE EXECUTION ERROR]: latexmk failed with exit code {e.returncode}."
        )
        if e.stdout:
            compile_log.append(f"latexmk STDOUT (on error):\n{e.stdout}")
        if e.stderr:
            compile_log.append(f"latexmk STDERR (on error):\n{e.stderr}")

        log_file_path = os.path.join(build_dir, "paper.log")
        if os.path.exists(log_file_path):
            try:
                with open(
                    log_file_path, "r", encoding="utf-8", errors="replace"
                ) as log_f:
                    compile_log.append(
                        f"\n--- Contents of paper.log ---\n{log_f.read()}"
                    )
            except Exception as e_log:
                compile_log.append(f"Could not read paper.log: {e_log}")
        return "\n".join(compile_log)
    except Exception as e_gen:
        compile_log.append(
            f"[CODE EXECUTION ERROR]: An unexpected error occurred: {str(e_gen)}"
        )
        return "\n".join(compile_log)


@tool
def generate_graph_from_dot(directory, dot_code, image_filename, format="png"):
    """
    Generates an image from DOT language code using the 'dot' command.

    Args:
        directory (str): The directory where the image file should be saved.
        dot_code (str): A string containing the DOT language code.
        image_filename (str): The desired name for the output image file.
        format (str): The output image format (e.g., 'png', 'svg', 'jpg'). Defaults to 'png'.

    Returns:
        str or None: The full path to the generated image file if successful,
                     otherwise None.
    """
    # Ensure the output directory exists
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return None

    output_filepath = os.path.join(directory, image_filename)

    command = ["dot", f"-T{format}", "-o", output_filepath]

    logger.debug(f"Attempting to run command: {' '.join(command)}")
    logger.debug(f"Saving output to: {output_filepath}")

    try:
        result = subprocess.run(
            command,
            input=dot_code,
            text=True,
            capture_output=True,
            check=True,
            encoding="utf-8",
        )

        logger.info("Command executed successfully.")

        return output_filepath

    except FileNotFoundError:
        logger.error("Error: 'dot' command not found.")
        logger.error(
            "Please ensure Graphviz is installed and accessible in your system's PATH."
        )
        logger.error("You can download it from https://graphviz.org/download/")
        logger.error("On Debian/Ubuntu: sudo apt-get install graphviz")
        logger.error("On macOS (using Homebrew): brew install graphviz")
        logger.error(
            "On Windows: Download installer or use winget install Graphviz.graphviz"
        )
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing dot command (Return code: {e.returncode}):")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Error output (stderr):\n{e.stderr}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


@tool
def generate_references_bib(lit_review_paper_content: str, lit_review_sum: str) -> str:
    """
    Generates a BibTeX (.bib) file content based on provided literature review papers and a summary.
    The BibTeX should primarily cite papers from `lit_review_paper`. If necessary,
    the LLM may suggest additional relevant citations (e.g., for datasets) not explicitly provided,
    and these must be in standard, compilable BibTeX format, clearly marked as LLM-generated in the title.

    Args:
        lit_review_paper_content (str): Full text or detailed information of papers in the literature review.
        lit_review_sum (str): A summary of the literature review.

    Returns:
        str: The content of the references.bib file in BibTeX format.
    """
    sys_prompt = """You are an expert academic assistant specializing in generating accurate and well-formatted BibTeX references.
Your task is to create the complete content for a `references.bib` file.

Instructions:
1.  **Prioritize:** Extract BibTeX entries from the `Provided Literature Review Papers` (full text or details). Focus on the most relevant papers.
2.  **Completeness:** Ensure each entry has essential fields relevant to its type (e.g., for @article: `author`, `title`, `journal`, `year`, `volume`, `number`, `pages`; for @inproceedings: `author`, `title`, `booktitle`, `year`, `pages`). Include `doi` or `url` when available.
3.  **Accuracy:** Cross-reference information with the `Literature Review Summary` for context and verify details.
4.  **Author Formatting:** In the `author` field, use the format "Firstname Lastname and Another Author" for multiple authors. **CRITICAL: Always use the word 'and' to separate authors, NEVER use '&' or any other symbols.**
5.  **LLM-Generated References (Optional but Strict):** If you believe a crucial, foundational reference (e.g., for a widely used dataset, a core algorithmic method, or a standard library) is absolutely necessary for the paper and is NOT explicitly in `Provided Literature Review Papers` (meaning, it's not present in the `lit_review_paper_content` you were given), you MAY add it.
    *   **CRITICAL: For any reference you add that was NOT explicitly in `Provided Literature Review Papers`, you MUST append `[LLM Generated]` to its `title` field within the BibTeX entry itself.** This ensures it appears in the final compiled PDF as LLM-generated.
    *   Example: `title={My New Concept [LLM Generated]}`
    *   **CRITICAL: Do NOT add any comments (lines starting with '%') within the BibTeX output.** The compiler expects clean BibTeX and comments are ignored in the final PDF output.
6.  **Output Format:** Output ONLY the pure BibTeX content. Do NOT include any introductory sentences, concluding remarks, or any text outside the BibTeX block.
7.  **Avoid Duplicates:** Ensure unique citation keys and avoid redundant entries.
8.  **Citation Keys:** Generate meaningful, concise citation keys (e.g., `AuthorYear` or `FirstAuthorYear`, e.g., `Smith2023`, `JonesEtAl2022`).
"""
    prompt = f"""
Provided Literature Review Papers (full text or details):
```text
{lit_review_paper_content}
```

Literature Review Summary (for context):
```text
{lit_review_sum}
```

Please generate the complete content for `references.bib`.
"""
    bib_content = query_model(prompt=prompt, system_prompt=sys_prompt)
    return bib_content


@tool
def propose_figures_and_content(paper_current_draft: str, research_plan: str) -> str:
    """
    Analyzes the current paper draft and research plan to propose necessary figures
    (beyond experimental result plots) and detailed content descriptions for each.
    This includes, but is not limited to, Architecture Diagrams, System Overviews, Flowcharts, etc.

    Args:
        paper_current_draft (str): The current LaTeX draft of the paper.
        research_plan (str): The original research plan and paper structure.

    Returns:
        str: A JSON string containing a list of proposed figures, each with a 'title'
             and 'description' of its content.
    """
    sys_prompt = """You are an expert academic assistant specializing in scientific visualization, paper structure, and **highly detailed diagram specification**.
    Your task is to analyze a research paper draft and its original plan to identify necessary figures. One to two figures are enough.
    For each proposed figure, you must provide a concise title and a **VERY DETAILED, CONCRETE description** of the content it should convey.

    **The 'description' you provide MUST be sufficient for another LLM (specializing in DOT code generation) to directly produce a diagram without further clarification.**
    Think of it as providing step-by-step instructions for drawing, including:

    **Key elements to include in 'description' for each figure:**
    -   **Main Components/Nodes:** List all primary entities, processes, or concepts (e.g., "Input Data", "Preprocessing Module", "IW-RF Model", "Output Prediction").
    -   **Relationships/Flows:** Describe how components connect, indicating direction (e.g., "Input Data feeds into Preprocessing", "Features are sampled from Cumulative Importance").
    -   **Sub-components/Internal Structure (if applicable):** For complex nodes, break down their internal parts (e.g., "Within the IW-RF Model: Decision Trees, Cumulative Importance Tracker, Feature Sampler").
    -   **Specific Operations/Actions (for flowcharts/process diagrams):** Detail the sequence of steps, decisions, and operations (e.g., "Step 1: Bootstrap Sample, Step 2: Select Features, Step 3: Grow Tree").
    -   **Data Types/Outputs (if relevant):** Specify what flows between components (e.g., "Outputs raw features", "Produces feature importances vector").
    -   **Any crucial labels, parameters, or annotations:** Mention specific text labels for nodes or edges that are vital for clarity (e.g., "Arrow labeled 'Weighted Sample'", "Node labeled 'Soft Split Probability'").
    -   **High-level structure/layout suggestions (optional but helpful):** Briefly mention if it should be a vertical flow, horizontal flow, tree-like, etc.

    **Figure types to propose:**
    -   **Architecture Diagram:** Illustrates the overall system, its components, and their interactions.
    -   **System Overview:** A high-level view of the entire system.
    -   **Flowchart:** Details a process or algorithm step-by-step.
    -   **Process Diagram:** Shows the sequence of operations or activities.
    -   **Conceptual Diagram:** Represents abstract ideas and relationships.
    -   **Schematic:** A detailed technical diagram.
    -   **Algorithm Pseudo-code Flow:** A visual representation of an algorithm's logic.

    Output MUST be a JSON list of objects, each with 'title' (str) and 'description' (str) fields.

    **Example for 'description' (for an Algorithm Flowchart):**
    "A flowchart detailing the steps of the proposed Importance-Weighted Random Forest (IW-RF) algorithm. Start with an 'Initialization' node for cumulative importance. Then, a loop for 'Each Tree b in Ensemble'. Inside the loop: 'Bootstrap Sample' -> 'Calculate Feature Probabilities' based on cumulative importance -> 'Sample Features' -> 'Grow Decision Tree'. After tree growth: 'Calculate Feature Importance Ib' from tree -> 'Update Cumulative Importance S_b' using decay formula. Arrows should connect steps with clear labels for data flow (e.g., 'X_sample, y_sample', 'Ib', 'S_b'). Decision points if any should be diamond-shaped."

    **Example for 'description' (for a System Architecture Diagram):**
    "A block diagram showing the data processing pipeline of the IW-RF model. Start with an 'Input Data' block. This feeds into a 'Data Preprocessing' module. The output then goes to the 'IW-RF Ensemble Model' block. The ensemble model produces 'Output Predictions'. Within the 'IW-RF Ensemble Model' block, show sub-blocks for 'Individual Decision Trees' and a central 'Feature Importance Tracker'. Arrows from 'Individual Decision Trees' feed into 'Feature Importance Tracker' (indicating importance calculation), and 'Feature Importance Tracker' feeds back into 'Individual Decision Trees' (indicating feature sampling guidance)."
    """
    prompt = f"""
Current Paper Draft (LaTeX):
```latex
{paper_current_draft}
```

Original Research Plan:
```text
{research_plan}
```

Based on the draft and plan, what figures (excluding experimental result plots) are necessary for this paper?
Provide a title and detailed content description for each.
"""
    response_json_str = query_model(prompt=prompt, system_prompt=sys_prompt)
    response_json_str = extract_prompt(response_json_str, "json")
    try:
        figures_proposal = json.loads(response_json_str)
        # Validate structure
        if not isinstance(figures_proposal, list) or not all(
            isinstance(f, dict) and "title" in f and "description" in f
            for f in figures_proposal
        ):
            raise ValueError("Invalid JSON structure for figures proposal.")
        return json.dumps(figures_proposal)  # Return validated JSON
    except json.JSONDecodeError:
        return f"Error: LLM returned invalid JSON for figure proposals: {response_json_str}"
    except ValueError as e:
        return f"Error: {e}. LLM returned: {response_json_str}"


@tool
def generate_dot_from_description(figure_title: str, figure_description: str) -> str:
    """
    Generates DOT language code for a figure based on its title and a detailed description.
    This DOT code can then be used by the `generate_graph_from_dot` tool.

    Args:
        figure_title (str): The title of the figure (e.g., "System Architecture").
        figure_description (str): A detailed description of the figure's content and components.

    Returns:
        str: A string containing the DOT language code for the figure.
    """
    sys_prompt = """You are an expert in Graphviz DOT language, specializing in generating **concise, visually clear, and aesthetically pleasing diagrams** from textual descriptions.
        Your primary goal is to transform the provided figure title and **highly detailed content description** into valid DOT code that is easy to understand at a glance, resembling professional flowcharts or architecture diagrams.

        **CRITICAL Output Standard: The generated DOT code MUST be:**
        1.  **Syntactically Correct:** Ensure the DOT code is valid and compiles without errors using `dot` command.
        2.  **Concise and Clear:** Nodes and edges should have **short, precise labels**. Avoid putting entire sentences into labels; summarize concepts.
        3.  **Visually Optimized:**
            *   **Overall Structure:** Use `digraph G { ... }` (or `graph G { ... }` for undirected graphs if appropriate, but usually `digraph`).
            *   **Direction:** Prefer `rankdir=LR` (Left to Right) for sequential flows or `rankdir=TB` (Top to Bottom) for hierarchical structures, choose what makes most sense for the description.
            *   **Nodes:** Use appropriate shapes (e.g., `box` for processes/modules, `ellipse` for start/end/data, `diamond` for decisions). Use `fillcolor`, `style=filled` for subtle color distinctions (e.g., lightgrey, peachpuff, skyblue) as hints suggest. Ensure consistency.
            *   **Edges:** Use `->` for directed flow. Add `label` for edges if the relationship isn't obvious. Consider `arrowhead=vee`, `arrowsize=0.8`.
            *   **Subgraphs (Clusters):** **Crucially, use `subgraph cluster_name { label="Cluster Title"; ... }` to group related nodes logically and create clear visual sections/tiers.** This is vital for showing hierarchy and avoiding clutter.
            *   **Layout:** Aim for minimal edge crossings and logical spacing. Use `rank=same` for nodes that should appear on the same horizontal/vertical level.
            *   **Fonts:** Keep fonts simple and readable (e.g., `fontname="Helvetica"`).
        4.  **Content Extraction:** Carefully extract the **main components, relationships, and distinct stages/groups** from the `figure_description`. Do NOT try to represent every single word from the description as a separate node if it leads to clutter. Abstract and summarize.

        **Output Format:** Output ONLY the pure DOT code. No introductory sentences, no explanations, no text outside the `digraph G { ... }` block. Ensure the first line is `digraph G {` (or `graph G {` if undirected) and the last line is `}`.
        """
    prompt = f"""
Figure Title: {figure_title}
Figure Description:
{figure_description}

Please generate the DOT code for this figure.
"""
    dot_code = query_model(prompt=prompt, system_prompt=sys_prompt)

    # Remove any markdown code block wrappers if the LLM adds them
    if dot_code.strip().startswith("```dot") and dot_code.strip().endswith("```"):
        dot_code = dot_code.strip()[len("```dot") : -len("```")].strip()
    elif dot_code.strip().startswith("```") and dot_code.strip().endswith("```"):
        # For general code block, try to remove it too
        dot_code = dot_code.strip()[len("```") : -len("```")].strip()

    return dot_code
