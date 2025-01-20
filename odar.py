#!/usr/bin/env python3
"""
On Device AI Reasoning (ODAR) - Enhanced DSPy Compiler
===================================================
A professional-grade compiler for Declarative Self-improving Python (DSPy),
with comprehensive error handling, logging, and configuration management.

Features:
    - Async support for improved performance
    - Comprehensive error handling and logging
    - Configuration file support
    - Enhanced prompt management
    - Sophisticated verification strategies
    - Metrics collection
    - Progress feedback

Usage:
    python odar.py <path_to_dspy_file> [--config CONFIG] [--debug] [--async]
"""

import sys
import json
import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from abc import ABC, abstractmethod

import aiohttp
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme
from rich.logging import RichHandler
from rich.table import Table

# Configure rich console with custom theme
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold"
})

console = Console(theme=CUSTOM_THEME)

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("odar")

@dataclass
class CompilerConfig:
    """Configuration settings for the DSPy compiler."""
    server_url: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    max_iterations: int
    timeout: int

    @classmethod
    def from_yaml(cls, path: Path) -> 'CompilerConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    @classmethod
    def default(cls) -> 'CompilerConfig':
        """Create default configuration."""
        return cls(
            server_url="http://localhost:8080/completion",
            max_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_iterations=3,
            timeout=300
        )

class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""
    
    @abstractmethod
    def create(self, question_data: dict) -> str:
        """Create a formatted prompt."""
        pass

class ReasoningPromptTemplate(PromptTemplate):
    """Standard reasoning prompt template with chat format and thinking structure."""
    
    def create(self, question_data: dict) -> str:
        base_template = """<|system|>
You are a helpful AI assistant that specializes in solving complex problems in {category}. 
You have extensive knowledge of mathematics, computer science, and engineering principles.
Please solve the following problem step by step, showing all your work and explaining your reasoning.

Question {id}: {question}

Please think through it step by step and show your reasoning process.
<|Assistant|>
I'll help you solve this {category} problem. Let me think about it carefully.

<think>
First, let me analyze what we're dealing with:
- Understand the key concepts and requirements
- Break down the problem into manageable steps
- Identify relevant formulas and principles
- Plan the solution approach
</think>

Let me solve this step by step:"""
        
        return base_template.format(**question_data)

class DSPyCompiler:
    """Enhanced DSPy compiler with async support and comprehensive error handling."""

    def __init__(self, config: CompilerConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        self.prompt_template = ReasoningPromptTemplate()
        self.metrics: Dict[str, Any] = {
            'iterations': 0,
            'total_tokens': 0,
            'verification_failures': 0
        }

    @lru_cache(maxsize=1000)
    def _cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return str(hash(prompt))

    async def read_dspy_file(self, filepath: Optional[Path]) -> list:
        """Asynchronously read DSPy code from file or stdin and parse questions."""
        try:
            if filepath:
                content = await asyncio.to_thread(Path(filepath).read_text, encoding='utf-8')
            else:
                content = await asyncio.to_thread(sys.stdin.read)
            
            # Parse JSON content
            data = json.loads(content)
            return data.get('questions', [])
            
        except Exception as e:
            logger.error(f"Error reading DSPy file: {e}")
            raise

    async def request_completion(
        self,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> Tuple[Optional[str], Optional[str]]:
        """Send prompt to LLM server and get response asynchronously."""
        params = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty
        }

        try:
            async with session.post(
                self.config.server_url,
                json=params,
                timeout=self.config.timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if self.debug:
                    logger.debug(f"Server response: {json.dumps(data, indent=2)}")

                content = self._extract_content(data)
                if content:
                    self.metrics['total_tokens'] += len(content.split())
                    return content, None
                return None, "No valid content in response"

        except Exception as e:
            logger.error(f"Error in completion request: {e}")
            return None, str(e)

    def _extract_content(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract content from server response."""
        if "content" in data:
            return data["content"]
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text")
        if "response" in data:
            return data["response"]
        return None

    async def verify_result(self, prompt: str, result: str) -> Tuple[bool, str]:
        """Verify the quality and completeness of the response."""
        if not result or len(result.strip()) == 0:
            return False, "Empty response"

        # Basic length check
        if len(result.split()) < 20:
            return False, "Response too short"

        # Check for mathematical content when LaTeX is present
        has_latex = '\\[' in result or '\\(' in result
        if has_latex:
            latex_indicators = ['\\pi', '\\times', '\\approx', '\\frac', '\\text']
            if any(indicator in result for indicator in latex_indicators):
                return True, "Valid mathematical response"

        # Check for step-by-step reasoning
        reasoning_indicators = [
            'step', 'first', 'then', 'finally', 'therefore', 'thus',
            'calculate', 'determine', 'find', 'solve'
        ]
        
        if any(indicator in result.lower() for indicator in reasoning_indicators):
            return True, "Valid reasoning process"

        return False, "No clear reasoning or mathematical process"

    def refine_prompt(self, prompt: str, iteration: int, verification_reason: str) -> str:
        """Create refined prompt based on verification feedback."""
        refinement_instructions = {
            "Empty response": "\nPlease provide a complete solution with detailed steps.",
            "Response too short": "\nPlease provide a more comprehensive explanation.",
            "No clear reasoning or mathematical process": "\nPlease show your work step by step, including calculations."
        }

        instruction = refinement_instructions.get(
            verification_reason,
            "\nPlease provide a detailed solution showing all steps and calculations."
        )

        refined_prompt = f"{prompt}\n\nAttempt #{iteration + 1}:{instruction}"
        
        if iteration > 0:
            # Add more specific guidance for subsequent attempts
            refined_prompt += "\nMake sure to:"
            refined_prompt += "\n- Show all mathematical steps clearly"
            refined_prompt += "\n- Include units in calculations"
            refined_prompt += "\n- Explain your reasoning at each step"
        
        return refined_prompt

    async def compile(self, question_data: dict) -> str:
        """Compile DSPy code with controlled concurrency and proper timeouts."""
        base_prompt = self.prompt_template.create(question_data)
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent requests
        
        async def try_single_compilation(prompt: str, session: aiohttp.ClientSession, progress_task_id) -> Tuple[Optional[str], bool]:
            """Try a single compilation with proper timeout and error handling."""
            try:
                async with semaphore:  # Control concurrency
                    result, error = await asyncio.wait_for(
                        self.request_completion(prompt, session),
                        timeout=30  # Hard timeout of 30 seconds
                    )
                    
                    if error:
                        logger.warning(f"Completion error: {error}")
                        return None, False
                    
                    if not result or len(result.strip()) == 0:
                        logger.warning("Empty response received")
                        return None, False
                        
                    is_acceptable, reason = await self.verify_result(prompt, result)
                    if is_acceptable:
                        self.metrics['iterations'] += 1
                        return result, True
                    
                    logger.info(f"Verification failed: {reason}")
                    self.metrics['iterations'] += 1
                    self.metrics['verification_failures'] += 1
                    return None, False
                    
            except asyncio.TimeoutError:
                logger.warning("Request timed out")
                return None, False
            except Exception as e:
                logger.error(f"Unexpected error during compilation: {e}")
                return None, False

        # Create a single progress display for the entire process
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        )

        try:
            async with aiohttp.ClientSession() as session:
                with progress:
                    # Add main compilation task
                    main_task = progress.add_task(
                        "[cyan]Compiling DSPy code...",
                        total=self.config.max_iterations
                    )

                    current_prompt = base_prompt
                    for i in range(self.config.max_iterations):
                        try:
                            result, success = await try_single_compilation(current_prompt, session, main_task)
                            progress.update(main_task, advance=1)
                            
                            if success and result:
                                return result
                            
                            # Only refine prompt if we haven't succeeded and haven't reached max iterations
                            if i < self.config.max_iterations - 1:
                                current_prompt = self.refine_prompt(base_prompt, i, "Previous attempt unsuccessful")
                                await asyncio.sleep(1)  # Add a small delay between attempts
                                
                        except Exception as e:
                            logger.error(f"Iteration {i} failed: {e}")
                            continue

            return "Compilation failed: Maximum iterations reached without success"
        except Exception as e:
            logger.error(f"Compilation process failed: {e}")
            return f"Compilation failed: {str(e)}"

async def main():
    """Main entry point with command line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced DSPy Compiler with async support and comprehensive error handling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dspy_file",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the .dspy file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--question",
        type=int,
        help="Process specific question number (1-based index)"
    )

    args = parser.parse_args()

    try:
        # Configure debug logging if requested
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Load configuration
        config = (CompilerConfig.from_yaml(args.config) 
                 if args.config else CompilerConfig.default())

        # Initialize compiler
        compiler = DSPyCompiler(config, debug=args.debug)

        # Read questions
        questions = await compiler.read_dspy_file(args.dspy_file)
        
        if not questions:
            console.print("[error]No questions found in the input file[/]")
            sys.exit(1)

        # Process single question if specified
        if args.question is not None:
            if 1 <= args.question <= len(questions):
                questions = [questions[args.question - 1]]
            else:
                console.print(f"[error]Invalid question number. Please specify a number between 1 and {len(questions)}[/]")
                sys.exit(1)

        # Process each question
        for question in questions:
            console.rule(f"[bold cyan]Question {question['id']}: {question['category']}[/]")
            
            # Compile code
            console.print("\n[info]Processing question...[/]")
            result = await compiler.compile(question)

            # Print results with proper formatting
            console.rule("[success]Solution[/]")
            console.print(result)

            # Print metrics in a table format
            console.rule("[info]Processing Metrics[/]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value")
            
            for key, value in compiler.metrics.items():
                table.add_row(
                    key.replace('_', ' ').title(),
                    str(value)
                )
            console.print(table)
            
            # Reset metrics for next question
            compiler.metrics = {
                'iterations': 0,
                'total_tokens': 0,
                'verification_failures': 0
            }
            
            # Add spacing between questions
            console.print("\n" + "="*80 + "\n")

    except Exception as e:
        logger.exception("[error]Processing failed[/]")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[warning]Compilation interrupted by user[/]")
        sys.exit(130)