"""
Patch generation and application system
"""

import os
import shutil
import tempfile
from typing import List, Dict, Any
from dataclasses import dataclass
from coda.adapters.llm_adapter import LLMAdapter
from coda.core.planner import StructuredPlan

@dataclass
class Patch:
    filename: str
    old_content: str
    new_content: str
    diff: str
    description: str

@dataclass
class PatchResult:
    filename: str
    success: bool
    error: str = None

class PatchManager:
    """Manages patch generation and application"""
    
    def __init__(self, config):
        self.config = config
        self.backup_dir = os.path.join(config.coda_dir, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def generate_patches(self, task_description: str, plan: StructuredPlan,
                        provider: str = 'openai', model: str = None) -> List[Patch]:
        """Generate patches based on the structured plan"""
        
        # For this MVP, we'll create a simplified patch generation
        # In a full implementation, this would iterate through plan steps
        
        patches = []
        
        # Get all unique files mentioned in the plan
        all_files = set()
        for step in plan.steps:
            all_files.update(step.files_involved)
        
        # Generate patches for each file
        for filename in all_files:
            if not filename or not os.path.exists(filename):
                continue
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    old_content = f.read()
                
                # Generate new content using LLM
                new_content = self._generate_file_content(
                    filename, old_content, task_description, plan, provider, model
                )
                
                if new_content and new_content != old_content:
                    diff = self._create_diff(filename, old_content, new_content)
                    
                    patch = Patch(
                        filename=filename,
                        old_content=old_content,
                        new_content=new_content,
                        diff=diff,
                        description=f"Modified {filename} for: {task_description}"
                    )
                    patches.append(patch)
            
            except Exception as e:
                print(f"Warning: Could not generate patch for {filename}: {e}")
        
        return patches
    
    def _generate_file_content(self, filename: str, old_content: str, 
                              task_description: str, plan: StructuredPlan,
                              provider: str, model: str) -> str:
        """Generate new file content using LLM"""
        
        from coda.adapters.llm_adapter import LLMAdapter
        llm_adapter = LLMAdapter()
        
        prompt = f"""You are modifying a file as part of implementing this task: {task_description}

Implementation Plan:
{plan.formatted_plan}

Current file content ({filename}):
```
{old_content}
```

Modify this file according to the task requirements and implementation plan.
Return ONLY the complete modified file content, with no explanations or markdown formatting.
Ensure the changes are minimal and focused on the specific requirements.
"""
        
        messages = [
            {"role": "system", "content": "You are an expert programmer. Return only the modified code with no explanations."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = llm_adapter.chat(messages, provider=provider, model=model)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating content for {filename}: {e}")
            return old_content
    
    def _create_diff(self, filename: str, old_content: str, new_content: str) -> str:
        """Create a unified diff string"""
        import difflib
        
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm=''
        )
        
        return ''.join(diff)
    
    def apply_patches(self, patches: List[Patch]) -> List[PatchResult]:
        """Apply patches with backup"""
        results = []
        
        for patch in patches:
            try:
                # Create backup
                backup_path = self._create_backup(patch.filename)
                
                # Apply patch
                with open(patch.filename, 'w', encoding='utf-8') as f:
                    f.write(patch.new_content)
                
                results.append(PatchResult(
                    filename=patch.filename,
                    success=True
                ))
                
            except Exception as e:
                results.append(PatchResult(
                    filename=patch.filename,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def _create_backup(self, filename: str) -> str:
        """Create backup of file before modification"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{filename.replace('/', '_')}_{timestamp}.backup"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(filename, backup_path)
        
        return backup_path