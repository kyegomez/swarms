#!/usr/bin/env python3
"""
Test script to verify autonomous loop streaming support.

This script tests that streaming works correctly in all three phases:
1. Planning Phase - should stream
2. Execution Phase - should stream
3. Summary Phase - should NOW stream (this was the fix)
"""

import sys
from swarms import Agent


def test_autonomous_loop_streaming():
    """Test autonomous loop with streaming enabled."""
    
    print("\n" + "="*80)
    print("TESTING AUTONOMOUS LOOP WITH STREAMING SUPPORT")
    print("="*80 + "\n")
    
    # Create a streaming callback to collect tokens
    collected_tokens = {
        "planning_phase": [],
        "execution_phase": [],
        "summary_phase": [],
    }
    
    current_phase = "planning_phase"
    
    def streaming_callback(token):
        """Callback to receive streaming tokens."""
        # Print token for real-time feedback
        if isinstance(token, dict):
            # Handle token info dict from call_llm
            if "token" in token:
                print(token["token"], end="", flush=True)
                collected_tokens[current_phase].append(token["token"])
        else:
            # Handle plain string token
            print(str(token), end="", flush=True)
            collected_tokens[current_phase].append(str(token))
    
    try:
        # Initialize agent with streaming enabled and max_loops="auto"
        agent = Agent(
            agent_name="StreamingTestAgent",
            system_prompt="""You are a helpful assistant that provides clear, comprehensive responses.
            When asked to plan, provide structured step-by-step plans.
            When asked to execute, provide detailed execution results.
            When asked to summarize, provide comprehensive summaries.""",
            model_name="gpt-4o-mini",
            max_loops="auto",
            streaming_on=True,
            verbose=False,  # Reduce noise
            print_on=False,
        )
        
        # Test task that will trigger autonomous loop
        task = """Please help me with the following:
        1. Create a plan for organizing a small tech conference
        2. Outline the key steps needed
        3. List important considerations
        
        Then provide a comprehensive summary of everything discussed."""
        
        print(f"📋 Task: {task}\n")
        print("-" * 80)
        print("🎬 EXECUTING WITH AUTONOMOUS LOOP STREAMING...\n")
        
        # Run agent with streaming callback
        result = agent.run(
            task=task,
            streaming_callback=streaming_callback,
        )
        
        print("\n" + "-" * 80)
        print("\n✅ EXECUTION COMPLETED!\n")
        
        # Verify results
        print("📊 STREAMING VERIFICATION REPORT:")
        print("-" * 80)
        
        phases_with_tokens = []
        phases_without_tokens = []
        
        for phase, tokens in collected_tokens.items():
            token_count = len(tokens)
            if token_count > 0:
                phases_with_tokens.append((phase, token_count))
                status = "✅"
            else:
                phases_without_tokens.append(phase)
                status = "❌"
            
            print(f"{status} {phase}: {token_count} tokens streamed")
        
        print("\n" + "="*80)
        if "summary_phase" in [p for p, _ in phases_with_tokens]:
            print("✅ SUCCESS: Summary phase is NOW streaming!")
            print("   The autonomous loop streaming fix is working correctly.")
        else:
            print("❌ ISSUE: Summary phase is still not streaming.")
            print("   The fix may need additional investigation.")
        
        print("="*80 + "\n")
        
        # Show final result
        print("📄 FINAL AGENT OUTPUT:")
        print("-" * 80)
        print(result)
        print("-" * 80 + "\n")
        
        return len(phases_without_tokens) == 0
        
    except Exception as e:
        print(f"\n❌ ERROR during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_autonomous_loop_streaming()
    sys.exit(0 if success else 1)
