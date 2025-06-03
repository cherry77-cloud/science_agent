"""
ScienceAgent Workflows Package

This package contains workflow orchestration classes for the ScienceAgent system.
"""

from .data_preparation_workflow import DataAnalysisWorkflow
# Commented out due to missing dependencies - uncomment after installing required packages
# from .experiment_workflow import ExperimentCodingWorkflow  
# from .paper_writing_workflow import PaperWritingWorkflow
# from .plan_formulation_workflow import PlanFormulationWorkflow
# from .main_workflow import AgentLab

__all__ = [
    'DataAnalysisWorkflow',
    # 'ExperimentCodingWorkflow', 
    # 'PaperWritingWorkflow',
    # 'PlanFormulationWorkflow',
    # 'AgentLab'
]
