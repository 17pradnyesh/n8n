# app/repositories/__init__.py
from .workflow_repository import WorkflowGenerator

# Create a single instance to be used throughout the app
workflow_generator = WorkflowGenerator()

# Export the generate_workflow_from_instructions function
def generate_workflow_from_instructions(instructions: str):
    return workflow_generator.generate_workflow(instructions)

__all__ = ['WorkflowGenerator', 'generate_workflow_from_instructions']
