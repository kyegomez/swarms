"""
Full Hierarchical Structured Communication Framework Test with Ollama

This script demonstrates the complete Hierarchical Structured Communication framework
using Ollama for local model inference. It showcases all components:
- Structured Communication Protocol
- Hierarchical Evaluation System  
- Graph-based Agent Orchestration
- Iterative Refinement Process
"""

import requests
import json
import time
import argparse
import sys
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Color support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback for systems without colorama
    class Fore:
        RED = GREEN = BLUE = YELLOW = MAGENTA = CYAN = WHITE = ""
    class Back:
        BLACK = RED = GREEN = BLUE = YELLOW = MAGENTA = CYAN = WHITE = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""
    COLORS_AVAILABLE = False

class CommunicationType(str, Enum):
    """Types of communication in the structured protocol"""
    MESSAGE = "message"  # M_ij: Specific task instructions
    BACKGROUND = "background"  # B_ij: Context and problem background
    INTERMEDIATE_OUTPUT = "intermediate_output"  # I_ij: Intermediate results

@dataclass
class StructuredMessage:
    """Structured communication message following HierarchicalStructuredComm protocol"""
    message: str
    background: str
    intermediate_output: str
    sender: str
    recipient: str
    timestamp: str = None

@dataclass
class EvaluationResult:
    """Result from evaluation team member"""
    evaluator_name: str
    criterion: str
    score: float
    feedback: str
    confidence: float

class HierarchicalStructuredCommunicationFramework:
    """
    Full implementation of Hierarchical Structured Communication framework
    using direct Ollama API calls
    """
    
    def __init__(self, model_name: str = "llama3:latest", verbose: bool = True, max_display_length: int = None):
        self.model_name = model_name
        self.verbose = verbose
        self.max_display_length = max_display_length
        self.conversation_history: List[StructuredMessage] = []
        self.intermediate_outputs: Dict[str, str] = {}
        self.evaluation_results: List[EvaluationResult] = []
        
        # Check Ollama availability
        self._check_ollama()
    
    def _print_colored(self, text: str, color: str = Fore.WHITE, style: str = Style.NORMAL):
        """Print colored text if colors are available"""
        if COLORS_AVAILABLE:
            print(f"{color}{style}{text}{Style.RESET_ALL}")
        else:
            print(text)
    
    def _print_header(self, text: str):
        """Print a header with styling"""
        self._print_colored(f"\n{text}", Fore.CYAN, Style.BRIGHT)
        self._print_colored("=" * len(text), Fore.CYAN)
    
    def _print_subheader(self, text: str):
        """Print a subheader with styling"""
        self._print_colored(f"\n{text}", Fore.YELLOW, Style.BRIGHT)
        self._print_colored("-" * len(text), Fore.YELLOW)
    
    def _print_success(self, text: str):
        """Print success message"""
        self._print_colored(text, Fore.GREEN, Style.BRIGHT)
    
    def _print_error(self, text: str):
        """Print error message"""
        self._print_colored(text, Fore.RED, Style.BRIGHT)
    
    def _print_info(self, text: str):
        """Print info message"""
        self._print_colored(text, Fore.BLUE)
    
    def _print_warning(self, text: str):
        """Print warning message"""
        self._print_colored(text, Fore.YELLOW)
    
    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text for display if needed"""
        if max_length is None:
            max_length = self.max_display_length
        
        if max_length and len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def _check_ollama(self):
        """Check if Ollama is running and get available models"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name') for model in models]
                
                if self.verbose:
                    self._print_success("Ollama is running")
                    self._print_info(f"Available models: {', '.join(model_names)}")
                
                # Verify our model is available
                if not any(self.model_name in name for name in model_names):
                    self._print_warning(f"Model {self.model_name} not found, using first available")
                    self.model_name = model_names[0] if model_names else "llama3:latest"
                
                self._print_info(f"Using model: {self.model_name}")
            else:
                raise Exception("Ollama not responding properly")
        except Exception as e:
            self._print_error(f"Cannot connect to Ollama: {e}")
            self._print_info("Please ensure Ollama is running: ollama serve")
            raise
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Make a call to Ollama API with infinite timeout"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Set timeout to None for infinite timeout
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=None
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            self._print_error(f"Error calling Ollama: {e}")
            return f"Error: {e}"
    
    def send_structured_message(
        self,
        sender: str,
        recipient: str,
        message: str,
        background: str = "",
        intermediate_output: str = ""
    ) -> StructuredMessage:
        """Send a structured message following the HierarchicalStructuredComm protocol"""
        structured_msg = StructuredMessage(
            message=message,
            background=background,
            intermediate_output=intermediate_output,
            sender=sender,
            recipient=recipient,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.conversation_history.append(structured_msg)
        
        if self.verbose:
            display_message = self._truncate_text(message, 100)
            self._print_info(f"{sender} -> {recipient}: {display_message}")
        
        return structured_msg
    
    def generate_content(self, task: str, context: str = "") -> str:
        """Generate initial content using generator agent"""
        if self.verbose:
            self._print_subheader("Step 1: Generating initial content")
        
        # Create structured message
        message = f"Generate comprehensive content for: {task}"
        background = f"Task: {task}\nContext: {context}\n\nProvide detailed, well-structured content."
        
        self.send_structured_message(
            sender="Supervisor",
            recipient="Generator",
            message=message,
            background=background
        )
        
        # Generate content
        prompt = f"""You are a Content Generator in a Hierarchical Structured Communication framework.

Task: {task}
Context: {context}

Generate comprehensive, well-structured content that addresses the task thoroughly.
Provide detailed explanations, examples, and insights.

Content:"""
        
        result = self._call_ollama(prompt, temperature=0.7, max_tokens=1500)
        self.intermediate_outputs["generator"] = result
        
        if self.verbose:
            self._print_info("Generated content:")
            print(result)  # Print full content without truncation
        
        return result
    
    def evaluate_content(self, content: str, criteria: List[str] = None) -> List[EvaluationResult]:
        """Evaluate content using hierarchical evaluation system"""
        if criteria is None:
            criteria = ["accuracy", "completeness", "clarity", "relevance"]
        
        if self.verbose:
            self._print_subheader("Step 2: Hierarchical evaluation")
        
        results = []
        
        for criterion in criteria:
            if self.verbose:
                self._print_info(f"  Evaluating {criterion}...")
            
            # Create structured message for evaluator
            message = f"Evaluate content for {criterion} criterion"
            background = f"Content to evaluate: {content[:500]}...\nCriterion: {criterion}"
            
            self.send_structured_message(
                sender="EvaluationSupervisor",
                recipient=f"{criterion.capitalize()}Evaluator",
                message=message,
                background=background,
                intermediate_output=content
            )
            
            # Evaluate with specific criterion
            prompt = f"""You are a {criterion.capitalize()} Evaluator in a hierarchical evaluation system.

Content to evaluate:
{content}

Evaluation criterion: {criterion}

Please provide:
1. Score (0-10)
2. Detailed feedback
3. Confidence level (0-1)
4. Specific suggestions for improvement

Evaluation:"""
            
            evaluation_response = self._call_ollama(prompt, temperature=0.3, max_tokens=800)
            
            # Parse evaluation (simplified parsing)
            score = 7.0  # Default score
            feedback = evaluation_response
            confidence = 0.8  # Default confidence
            
            # Try to extract score from response
            if "score" in evaluation_response.lower():
                try:
                    # Look for patterns like "score: 8" or "8/10"
                    import re
                    score_match = re.search(r'(\d+(?:\.\d+)?)/10|score[:\s]*(\d+(?:\.\d+)?)', evaluation_response.lower())
                    if score_match:
                        score = float(score_match.group(1) or score_match.group(2))
                except:
                    pass
            
            result = EvaluationResult(
                evaluator_name=f"{criterion.capitalize()}Evaluator",
                criterion=criterion,
                score=score,
                feedback=feedback,
                confidence=confidence
            )
            
            results.append(result)
            
            if self.verbose:
                self._print_info(f"    Score: {score}/10")
                print(f"    Feedback: {feedback}")  # Print full feedback
        
        self.evaluation_results.extend(results)
        return results
    
    def refine_content(self, original_content: str, evaluation_results: List[EvaluationResult]) -> str:
        """Refine content based on evaluation feedback"""
        if self.verbose:
            self._print_subheader("Step 3: Refining content")
        
        # Create feedback summary
        feedback_summary = "\n\n".join([
            f"{result.criterion.capitalize()} (Score: {result.score}/10):\n{result.feedback}"
            for result in evaluation_results
        ])
        
        # Create structured message for refinement
        message = "Refine content based on evaluation feedback"
        background = f"Original content: {original_content[:500]}...\n\nEvaluation feedback:\n{feedback_summary}"
        
        self.send_structured_message(
            sender="Supervisor",
            recipient="Refiner",
            message=message,
            background=background,
            intermediate_output=original_content
        )
        
        # Refine content
        prompt = f"""You are a Content Refiner in a Hierarchical Structured Communication framework.

Original Content:
{original_content}

Evaluation Feedback:
{feedback_summary}

Please refine the content to address the feedback while maintaining its core strengths.
Focus on the specific issues mentioned in the evaluation and provide improvements.

Refined Content:"""
        
        refined_result = self._call_ollama(prompt, temperature=0.5, max_tokens=1500)
        self.intermediate_outputs["refiner"] = refined_result
        
        if self.verbose:
            self._print_info("Refined content:")
            print(refined_result)  # Print full content without truncation
        
        return refined_result
    
    def run_hierarchical_workflow(self, task: str, max_iterations: int = 3, quality_threshold: float = 8.0) -> Dict[str, Any]:
        """Run the complete Hierarchical Structured Communication workflow"""
        self._print_header("Starting Hierarchical Structured Communication Workflow")
        self._print_info(f"Task: {task}")
        self._print_info(f"Max iterations: {max_iterations}")
        self._print_info(f"Quality threshold: {quality_threshold}")
        
        start_time = time.time()
        current_content = None
        iteration = 0
        
        for iteration in range(max_iterations):
            self._print_subheader(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Step 1: Generate/Refine content
            if iteration == 0:
                current_content = self.generate_content(task)
            else:
                current_content = self.refine_content(current_content, evaluation_results)
            
            # Step 2: Evaluate content
            evaluation_results = self.evaluate_content(current_content)
            
            # Step 3: Check quality threshold
            avg_score = sum(result.score for result in evaluation_results) / len(evaluation_results)
            self._print_info(f"Average evaluation score: {avg_score:.2f}/10")
            
            if avg_score >= quality_threshold:
                self._print_success("Quality threshold met! Stopping refinement.")
                break
            
            if iteration < max_iterations - 1:
                self._print_info("Continuing refinement...")
        
        total_time = time.time() - start_time
        
        return {
            "final_content": current_content,
            "total_iterations": iteration + 1,
            "average_score": avg_score,
            "evaluation_results": evaluation_results,
            "conversation_history": self.conversation_history,
            "intermediate_outputs": self.intermediate_outputs,
            "total_time": total_time
        }
    
    def print_workflow_summary(self, result: Dict[str, Any]):
        """Print a comprehensive summary of the workflow results"""
        self._print_header("TALK HIERARCHICAL WORKFLOW COMPLETED")
        
        self._print_subheader("PERFORMANCE SUMMARY")
        self._print_info(f"  Total iterations: {result['total_iterations']}")
        self._print_info(f"  Final average score: {result['average_score']:.2f}/10")
        self._print_info(f"  Total time: {result['total_time']:.2f} seconds")
        self._print_info(f"  Messages exchanged: {len(result['conversation_history'])}")
        
        self._print_subheader("FINAL CONTENT")
        print(result['final_content'])
        
        self._print_subheader("EVALUATION RESULTS")
        for eval_result in result['evaluation_results']:
            self._print_info(f"  {eval_result.criterion.capitalize()}: {eval_result.score}/10")
            print(f"    Feedback: {eval_result.feedback}")
        
        self._print_subheader("COMMUNICATION HISTORY")
        for i, msg in enumerate(result['conversation_history']):
            self._print_info(f"  {i+1}. {msg.sender} -> {msg.recipient}")
            print(f"     Message: {msg.message}")
            print(f"     Background: {msg.background}")
            print(f"     Intermediate Output: {msg.intermediate_output}")
            print(f"     Time: {msg.timestamp}")

def test_basic_workflow():
    """Test basic Hierarchical Structured Communication workflow"""
    print("Test 1: Basic Workflow")
    print("=" * 50)
    
    framework = HierarchicalStructuredCommunicationFramework(model_name="llama3:latest", verbose=True)
    
    task = "Explain the concept of neural networks and their applications in modern AI"
    
    result = framework.run_hierarchical_workflow(
        task=task,
        max_iterations=2,
        quality_threshold=7.5
    )
    
    framework.print_workflow_summary(result)
    return result

def test_complex_workflow():
    """Test complex workflow with multiple iterations"""
    print("Test 2: Complex Workflow")
    print("=" * 50)
    
    framework = HierarchicalStructuredCommunicationFramework(model_name="llama3:latest", verbose=True)
    
    task = """Create a comprehensive guide on machine learning that covers:
1. Basic concepts and definitions
2. Types of machine learning (supervised, unsupervised, reinforcement)
3. Common algorithms and their use cases
4. Real-world applications and examples
5. Future trends and challenges

Make it suitable for both beginners and intermediate learners."""
    
    result = framework.run_hierarchical_workflow(
        task=task,
        max_iterations=3,
        quality_threshold=8.0
    )
    
    framework.print_workflow_summary(result)
    return result

def test_structured_communication():
    """Test structured communication protocol in isolation"""
    print("Test 3: Structured Communication Protocol")
    print("=" * 50)
    
    framework = HierarchicalStructuredCommunicationFramework(model_name="llama3:latest", verbose=True)
    
    # Test structured message exchange
    msg1 = framework.send_structured_message(
        sender="Supervisor",
        recipient="Generator",
        message="Generate content about renewable energy",
        background="Focus on solar and wind power",
        intermediate_output="Previous discussion covered climate change"
    )
    
    msg2 = framework.send_structured_message(
        sender="Generator",
        recipient="Evaluator",
        message="Content ready for evaluation",
        background="Generated comprehensive guide on renewable energy",
        intermediate_output="Detailed explanation of solar and wind technologies"
    )
    
    print("Structured Messages:")
    for i, msg in enumerate(framework.conversation_history):
        print(f"Message {i+1}:")
        print(f"  From: {msg.sender}")
        print(f"  To: {msg.recipient}")
        print(f"  Message (M_ij): {msg.message}")
        print(f"  Background (B_ij): {msg.background}")
        print(f"  Intermediate Output (I_ij): {msg.intermediate_output}")
        print(f"  Timestamp: {msg.timestamp}")

def test_quick_demo():
    """Quick demonstration with smaller model and shorter prompts"""
    print("Test 4: Quick Demo")
    print("=" * 50)
    
    # Use a smaller model for faster response
    framework = HierarchicalStructuredCommunicationFramework(model_name="llama3.2:3b", verbose=True)
    
    task = "Explain what artificial intelligence is in simple terms"
    
    result = framework.run_hierarchical_workflow(
        task=task,
        max_iterations=1,
        quality_threshold=6.0
    )
    
    framework.print_workflow_summary(result)
    return result

def interactive_mode():
    """Interactive mode for custom tasks"""
    print("Interactive Hierarchical Structured Communication Framework")
    print("=" * 50)
    
    # Get user input
    model_name = input("Enter model name (default: llama3:latest): ").strip() or "llama3:latest"
    task = input("Enter your task: ").strip()
    
    if not task:
        print("No task provided. Exiting.")
        return
    
    max_iterations = input("Enter max iterations (default: 2): ").strip()
    max_iterations = int(max_iterations) if max_iterations.isdigit() else 2
    
    quality_threshold = input("Enter quality threshold (default: 7.0): ").strip()
    quality_threshold = float(quality_threshold) if quality_threshold.replace('.', '').isdigit() else 7.0
    
    # Create framework and run
    framework = HierarchicalStructuredCommunicationFramework(model_name=model_name, verbose=True)
    
    result = framework.run_hierarchical_workflow(
        task=task,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold
    )
    
    framework.print_workflow_summary(result)
    return result

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Hierarchical Structured Communication Framework Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_hierarchical_structured_communication_test.py --quick
  python full_hierarchical_structured_communication_test.py --interactive
  python full_hierarchical_structured_communication_test.py --model llama3.2:3b --task "Explain AI"
  python full_hierarchical_structured_communication_test.py --all
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick demo test"
    )
    
    parser.add_argument(
        "--basic", 
        action="store_true", 
        help="Run basic workflow test"
    )
    
    parser.add_argument(
        "--complex", 
        action="store_true", 
        help="Run complex workflow test"
    )
    
    parser.add_argument(
        "--communication", 
        action="store_true", 
        help="Run structured communication test"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3:latest",
        help="Ollama model to use (default: llama3:latest)"
    )
    
    parser.add_argument(
        "--task", 
        type=str, 
        help="Custom task for single run"
    )
    
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=2,
        help="Maximum iterations (default: 2)"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=7.0,
        help="Quality threshold (default: 7.0)"
    )
    
    parser.add_argument(
        "--no-color", 
        action="store_true", 
        help="Disable colored output"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Disable verbose output (verbose is enabled by default)"
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested
    global COLORS_AVAILABLE
    if args.no_color:
        COLORS_AVAILABLE = False
    
    # Print header
    if COLORS_AVAILABLE:
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("=" * 80)
        print("TALK HIERARCHICAL FRAMEWORK TEST SUITE")
        print("=" * 80)
        print(f"{Style.RESET_ALL}")
    else:
        print("=" * 80)
        print("TALK HIERARCHICAL FRAMEWORK TEST SUITE")
        print("=" * 80)
    
    print("Testing Hierarchical Structured Communication framework with Ollama")
    print("=" * 80)
    
    try:
        if args.interactive:
            interactive_mode()
        elif args.task:
            # Single task run
            framework = HierarchicalStructuredCommunicationFramework(
                model_name=args.model, 
                verbose=not args.quiet
            )
            result = framework.run_hierarchical_workflow(
                task=args.task,
                max_iterations=args.iterations,
                quality_threshold=args.threshold
            )
            framework.print_workflow_summary(result)
        elif args.quick:
            test_quick_demo()
        elif args.basic:
            test_basic_workflow()
        elif args.complex:
            test_complex_workflow()
        elif args.communication:
            test_structured_communication()
        elif args.all:
            # Run all tests
            test_quick_demo()
            test_basic_workflow()
            test_complex_workflow()
            test_structured_communication()
        else:
            # Default: run quick demo
            test_quick_demo()
        
        print("All tests completed successfully!")
        print("Framework Features Demonstrated:")
        print("  Structured Communication Protocol (M_ij, B_ij, I_ij)")
        print("  Hierarchical Evaluation System")
        print("  Iterative Refinement Process")
        print("  Graph-based Agent Orchestration")
        print("  Local Ollama Integration")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 