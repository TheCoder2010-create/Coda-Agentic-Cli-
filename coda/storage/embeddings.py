"""
Embedding management and vector storage using scikit-learn
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
# Simplified version without numpy/sklearn to avoid dependency issues
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# Optional imports with fallbacks
try:
    from pathspec import PathSpec
    HAS_PATHSPEC = True
except ImportError:
    HAS_PATHSPEC = False
    
try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False

class EmbeddingManager:
    """Manages embeddings and vector storage for repository content"""
    
    def __init__(self, config):
        self.config = config
        self.embeddings_dir = config.get('storage.embeddings_dir')
        self.chunk_size = config.get('embeddings.chunk_size', 1000)
        self.chunk_overlap = config.get('embeddings.chunk_overlap', 200)
        
        # Simplified storage for documents (no vectorization for now)
        self.documents = []
        self.file_chunks = []
        
        # Load existing embeddings if available
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load existing embeddings from disk"""
        docs_path = os.path.join(self.embeddings_dir, 'documents.json')
        chunks_path = os.path.join(self.embeddings_dir, 'chunks.json')
        vectors_path = os.path.join(self.embeddings_dir, 'vectors.pkl')
        vectorizer_path = os.path.join(self.embeddings_dir, 'vectorizer.pkl')
        
        try:
            if all(os.path.exists(p) for p in [docs_path, chunks_path]):
                with open(docs_path, 'r') as f:
                    self.documents = json.load(f)
                
                with open(chunks_path, 'r') as f:
                    self.file_chunks = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing embeddings: {e}")
    
    def _save_embeddings(self):
        """Save embeddings to disk"""
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        docs_path = os.path.join(self.embeddings_dir, 'documents.json')
        chunks_path = os.path.join(self.embeddings_dir, 'chunks.json')
        
        with open(docs_path, 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        with open(chunks_path, 'w') as f:
            json.dump(self.file_chunks, f, indent=2)
    
    def _get_gitignore_patterns(self, repo_path: str):
        """Get gitignore patterns for the repository"""
        gitignore_path = os.path.join(repo_path, '.gitignore')
        patterns = [
            # Default patterns
            '.git/*',
            '__pycache__/*',
            '*.pyc',
            '.coda/*',
            'node_modules/*',
            '.env',
            '*.log'
        ]
        
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                patterns.extend(line.strip() for line in f if line.strip() and not line.startswith('#'))
        
        if HAS_PATHSPEC:
            return PathSpec.from_lines('gitwildmatch', patterns)
        else:
            # Simple fallback without pathspec
            return patterns
    
    def _chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            chunks.append({
                'content': text,
                'filename': filename,
                'start_word': 0,
                'end_word': len(words)
            })
        else:
            start = 0
            chunk_id = 0
            
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                
                chunks.append({
                    'content': ' '.join(chunk_words),
                    'filename': filename,
                    'start_word': start,
                    'end_word': end,
                    'chunk_id': chunk_id
                })
                
                # Move start position with overlap
                start = max(start + self.chunk_size - self.chunk_overlap, start + 1)
                chunk_id += 1
        
        return chunks
    
    def _is_text_file(self, filepath: str) -> bool:
        """Check if file is likely a text file"""
        text_extensions = {
            '.py', '.js', '.ts', '.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml',
            '.xml', '.csv', '.sql', '.sh', '.bash', '.zsh', '.fish', '.dockerfile',
            '.gitignore', '.env.example', '.toml', '.ini', '.cfg', '.conf'
        }
        
        ext = Path(filepath).suffix.lower()
        if ext in text_extensions:
            return True
        
        # Check if file has no extension but might be text
        if not ext:
            try:
                with open(filepath, 'rb') as f:
                    chunk = f.read(512)
                    # Simple heuristic: if most bytes are printable ASCII, it's probably text
                    printable_ratio = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in [9, 10, 13]) / len(chunk)
                    return printable_ratio > 0.7
            except:
                return False
        
        return False
    
    def index_repository(self, repo_path: str = '.', force: bool = False, 
                        exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Index repository files into embeddings"""
        repo_path = os.path.abspath(repo_path)
        
        # Get ignore patterns
        ignore_patterns = self._get_gitignore_patterns(repo_path)
        if exclude_patterns:
            ignore_patterns.extend(exclude_patterns)
        
        # Clear existing data if force rebuild
        if force:
            self.documents = []
            self.file_chunks = []
            self.vectors = None
        
        # Collect text files
        new_documents = []
        new_chunks = []
        total_size = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Filter directories
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.relpath(os.path.join(root, d), repo_path), ignore_patterns)]
            
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, repo_path)
                
                # Skip if matches ignore patterns
                if self._should_ignore(rel_path, ignore_patterns):
                    continue
                
                # Skip if not a text file
                if not self._is_text_file(filepath):
                    continue
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Skip empty files
                    if not content.strip():
                        continue
                    
                    # Add to documents
                    doc = {
                        'filename': rel_path,
                        'content': content,
                        'size': len(content)
                    }
                    new_documents.append(doc)
                    total_size += len(content)
                    
                    # Create chunks
                    chunks = self._chunk_text(content, rel_path)
                    new_chunks.extend(chunks)
                    
                except Exception as e:
                    print(f"Warning: Could not read {rel_path}: {e}")
                    continue
        
        # Update storage
        self.documents.extend(new_documents)
        self.file_chunks.extend(new_chunks)
        
        # Simplified indexing without vectors for now
        # In full implementation, this would create embeddings
        
        # Save to disk
        self._save_embeddings()
        
        return {
            'files_processed': len(new_documents),
            'chunks_created': len(new_chunks),
            'embeddings_stored': len(self.file_chunks),
            'total_size_mb': total_size / (1024 * 1024)
        }
    
    def search_relevant_context(self, query: str, max_tokens: int = 8000, 
                               top_k: int = 10) -> str:
        """Search for relevant context based on query (simplified version)"""
        if not self.file_chunks:
            return "No repository index found. Run 'coda index' first."
        
        # Simplified search - just return first few chunks that contain query terms
        query_terms = query.lower().split()
        relevant_chunks = []
        
        for chunk in self.file_chunks:
            content_lower = chunk['content'].lower()
            if any(term in content_lower for term in query_terms):
                relevant_chunks.append(chunk)
            
            if len(relevant_chunks) >= top_k:
                break
        
        # If no matches, return first few chunks
        if not relevant_chunks:
            relevant_chunks = self.file_chunks[:min(top_k, len(self.file_chunks))]
        
        # Build context within token limit
        context_parts = []
        current_tokens = 0
        
        for chunk in relevant_chunks:
            chunk_text = f"File: {chunk['filename']}\n{chunk['content']}\n"
            
            # Rough token estimation
            chunk_tokens = len(chunk_text.split()) * 1.3
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        return "\n" + "="*80 + "\n".join(context_parts)
    
    def _should_ignore(self, path: str, patterns: list) -> bool:
        """Simple pattern matching without pathspec dependency"""
        import fnmatch
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"*/{pattern}"):
                return True
        return False
    
    def get_file_context(self, filenames: List[str]) -> str:
        """Get context for specific files"""
        context_parts = []
        
        for filename in filenames:
            # Find matching documents
            matching_docs = [doc for doc in self.documents if doc['filename'] == filename]
            
            if matching_docs:
                doc = matching_docs[0]
                context_parts.append(f"File: {doc['filename']}\n{doc['content']}\n")
            else:
                # Try to read file directly if not indexed
                try:
                    if os.path.exists(filename):
                        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        context_parts.append(f"File: {filename}\n{content}\n")
                except Exception as e:
                    context_parts.append(f"File: {filename}\nError reading file: {e}\n")
        
        return "\n" + "="*80 + "\n".join(context_parts)