#!/usr/bin/env python3
"""
On Device AI Reasoning (ODAR) - Enhanced DSPy Compiler
===================================================
A professional-grade compiler for Declarative Self-improving Python (DSPy),
with comprehensive error handling, logging, and configuration management.
"""

import sys
import json
import asyncio
import logging
import argparse
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
    max_tokens: int = 32000  # Increased for better context
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 100
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.1
    max_iterations: int = 5
    timeout: int = 600
    batch_size: int = 10
    connection_limit: int = 100
    max_retries: int = 3

    @classmethod
    def from_yaml(cls, path: Path) -> 'CompilerConfig':
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return cls(**config)
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            return cls.default()

    @classmethod
    def default(cls) -> 'CompilerConfig':
        """Create default configuration."""
        return cls(
            server_url="http://localhost:8080/completion"
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
        base_template = """<|User|>
You are a highly advanced AI assistant specializing in {category} problems, with extensive knowledge 
of mathematics, computer science, and engineering principles. Please solve this complex problem 
using optimal reasoning strategies and mathematical rigor.

Question {id}: {question}

Approach this systematically with advanced problem-solving techniques.
<|Assistant|>
I'll solve this {category} problem using advanced analytical methods.

<think>
Analysis framework:
1. Core concept identification and problem space mapping
2. Mathematical principle application and optimization
3. Solution strategy development with complexity analysis
4. Implementation planning with efficiency considerations
</think>

Detailed solution process:"""
        
        return base_template.format(**question_data)

class DSPyCompiler:
    """Enhanced DSPy compiler with async support and comprehensive error handling."""

    def __init__(self, config: CompilerConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        self.prompt_template = ReasoningPromptTemplate()
        self.metrics = {
            'iterations': 0,
            'total_tokens': 0,
            'verification_failures': 0,
            'cache_hits': 0
        }
        self._session = None

    async def initialize(self):
        """Initialize aiohttp session with connection pooling."""
        if not self._session:
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_limit,
                force_close=False,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

    async def cleanup(self):
        """Cleanup resources."""
        if self._session:
            await self._session.close()

    @lru_cache(maxsize=1000)
    def _cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return str(hash(prompt))

    async def read_dspy_file(self, filepath: Optional[Path]) -> list:
        """Read DSPy code from file or stdin and parse questions."""
        try:
            if filepath:
                content = await asyncio.to_thread(Path(filepath).read_text, encoding='utf-8')
            else:
                content = await asyncio.to_thread(sys.stdin.read)
            
            data = json.loads(content)
            return data.get('questions', [])
            
        except Exception as e:
            logger.error(f"Error reading DSPy file: {e}")
            raise

    async def request_completion(
        self,
        prompt: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Send prompt to LLM server and get response asynchronously."""
        if not self._session:
            await self.initialize()

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

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(
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
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    return None, str(e)
                await asyncio.sleep(1)  # Wait before retry

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

        if len(result.split()) < 50:  # Increased minimum length requirement
            return False, "Response too short"

        # Check for mathematical content when LaTeX is present
        has_latex = '\\[' in result or '\\(' in result
        if has_latex:
            latex_indicators = ['\\pi', '\\times', '\\approx', '\\frac', '\\text', '\\sum', '\\int']
            if any(indicator in result for indicator in latex_indicators):
                return True, "Valid mathematical response"

        # Check for step-by-step reasoning
        reasoning_indicators = [
            'step', 'first', 'then', 'finally', 'therefore', 'thus',
            'calculate', 'determine', 'find', 'solve', 'analyze', 'conclude'
        ]
        
        if any(indicator in result.lower() for indicator in reasoning_indicators):
            return True, "Valid reasoning process"

        return False, "No clear reasoning or mathematical process"

    async def compile(self, question_data: dict) -> str:
        """Compile DSPy code with enhanced error handling and retries."""
        try:
            base_prompt = self.prompt_template.create(question_data)
            current_prompt = base_prompt

            for iteration in range(self.config.max_iterations):
                result, error = await self.request_completion(current_prompt)
                
                if error:
                    logger.warning(f"Completion error: {error}")
                    if iteration == self.config.max_iterations - 1:
                        return f"Compilation failed: {error}"
                    continue

                is_valid, reason = await self.verify_result(current_prompt, result)
                if is_valid:
                    self.metrics['iterations'] += 1
                    return result

                logger.info(f"Verification failed: {reason}")
                self.metrics['verification_failures'] += 1
                
                # Refine prompt for next iteration
                current_prompt = f"{base_prompt}\n\nPrevious attempt was insufficient. Please provide a more detailed solution with clear steps and reasoning."

            return "Compilation failed: Maximum iterations reached without success"

        except Exception as e:
            logger.error(f"Compilation process failed: {e}")
            return f"Compilation failed: {str(e)}"

async def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced DSPy Compiler with async support and comprehensive error handling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dspy_file",
        nargs="?",
        type=Path,
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
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        config = (CompilerConfig.from_yaml(args.config) 
                 if args.config else CompilerConfig.default())

        compiler = DSPyCompiler(config, debug=args.debug)
        await compiler.initialize()

        try:
            questions = await compiler.read_dspy_file(args.dspy_file)
            
            if not questions:
                console.print("[error]No questions found in the input file[/]")
                sys.exit(1)

            if args.question is not None:
                if 1 <= args.question <= len(questions):
                    questions = [questions[args.question - 1]]
                else:
                    console.print(f"[error]Invalid question number. Please specify a number between 1 and {len(questions)}[/]")
                    sys.exit(1)

            for question in questions:
                result = await compiler.compile(question)
                
                console.rule("[success]Solution[/]")
                console.print(result)

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

                compiler.metrics = {
                    'iterations': 0,
                    'total_tokens': 0,
                    'verification_failures': 0,
                    'cache_hits': 0
                }

                console.print("\n" + "="*80 + "\n")

        finally:
            await compiler.cleanup()

    except Exception as e:
        logger.exception("[error]Processing failed[/]")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[warning]Compilation interrupted by user[/]")
        sys.exit(130)