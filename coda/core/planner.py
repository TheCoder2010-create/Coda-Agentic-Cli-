"""
Structured planning system for code changes
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from coda.adapters.llm_adapter import LLMAdapter
from coda.storage.embeddings import EmbeddingManager

@dataclass
class PlanStep:
    id: int
    description: str
    files_involved: List[str]
    estimated_effort: str
    dependencies: List[int]

@dataclass
class StructuredPlan:
    task_description: str
    steps: List[PlanStep]
    estimated_time: str
    complexity: str
    formatted_plan: str

class StructuredPlanner:
    """Creates structured plans for code changes"""
    
    def __init__(self, llm_adapter: LLMAdapter, embedding_manager: EmbeddingManager):
        self.llm_adapter = llm_adapter
        self.embedding_manager = embedding_manager
    
    def create_plan(self, task_description: str, provider: str = 'openai', 
                   model: str = None) -> StructuredPlan:
        """Create a structured plan for the given task"""
        
        # Get relevant context
        context = self.embedding_manager.search_relevant_context(
            task_description, max_tokens=6000
        )
        
        # Create planning prompt
        planning_prompt = f"""You are a senior software engineer creating a detailed implementation plan.

Task: {task_description}

Repository Context:
{context}

Create a structured implementation plan with the following format:

IMPLEMENTATION PLAN

## Overview
Brief description of the overall approach and strategy.

## Complexity Assessment
Rate complexity as: Low/Medium/High
Estimated time: X hours/days

## Step-by-Step Plan

1. [Step Title]
   - Description: What needs to be done
   - Files: List of files that will be modified/created
   - Effort: Low/Medium/High
   - Dependencies: None or list of previous steps

2. [Next Step]
   ...

## Risk Assessment
- Potential issues to watch for
- Testing requirements
- Rollback considerations

## Success Criteria
- How to verify the implementation is complete and correct

Provide a detailed, actionable plan that a developer can follow step by step.
"""
        
        messages = [
            {"role": "system", "content": "You are an expert software architect and project planner."},
            {"role": "user", "content": planning_prompt}
        ]
        
        response = self.llm_adapter.chat(messages, provider=provider, model=model)
        
        # Parse the response to extract structured information
        plan_steps = self._parse_plan_steps(response.content)
        
        return StructuredPlan(
            task_description=task_description,
            steps=plan_steps,
            estimated_time=self._extract_estimated_time(response.content),
            complexity=self._extract_complexity(response.content),
            formatted_plan=response.content
        )
    
    def _parse_plan_steps(self, plan_text: str) -> List[PlanStep]:
        """Parse plan text to extract structured steps"""
        steps = []
        lines = plan_text.split('\n')
        
        current_step = None
        step_id = 0
        
        for line in lines:
            line = line.strip()
            
            # Look for numbered steps
            if line and (line[0].isdigit() or line.startswith('Step')):
                if current_step:
                    steps.append(current_step)
                
                step_id += 1
                current_step = PlanStep(
                    id=step_id,
                    description=line,
                    files_involved=[],
                    estimated_effort="Medium",
                    dependencies=[]
                )
            
            elif current_step and line.lower().startswith('- files:'):
                # Extract file names
                files_text = line.split(':', 1)[1].strip()
                current_step.files_involved = [f.strip() for f in files_text.split(',') if f.strip()]
            
            elif current_step and line.lower().startswith('- effort:'):
                effort_text = line.split(':', 1)[1].strip().lower()
                if 'low' in effort_text:
                    current_step.estimated_effort = "Low"
                elif 'high' in effort_text:
                    current_step.estimated_effort = "High"
                else:
                    current_step.estimated_effort = "Medium"
        
        if current_step:
            steps.append(current_step)
        
        return steps
    
    def _extract_estimated_time(self, plan_text: str) -> str:
        """Extract estimated time from plan text"""
        lines = plan_text.lower().split('\n')
        for line in lines:
            if 'estimated time' in line or 'time:' in line:
                return line.split(':')[-1].strip()
        return "Unknown"
    
    def _extract_complexity(self, plan_text: str) -> str:
        """Extract complexity assessment from plan text"""
        lines = plan_text.lower().split('\n')
        for line in lines:
            if 'complexity' in line:
                if 'low' in line:
                    return "Low"
                elif 'high' in line:
                    return "High"
                elif 'medium' in line:
                    return "Medium"
        return "Medium"