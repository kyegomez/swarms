import time
from typing import List

from swarms import Agent

from examples.multi_agent.simulations.agent_map.agent_map_simulation import (
    AgentMapSimulation,
)

# Create a natural conversation prompt for the simulation
NATURAL_CONVERSATION_PROMPT = """

You are a friendly, knowledgeable professional who enjoys meeting and talking with colleagues. 

When you meet someone new:
- Be warm and personable in your interactions
- Keep your responses conversational and relatively brief (1-3 sentences typically)
- Ask thoughtful questions and show genuine interest in what others have to say
- Share your expertise naturally without being overly formal or academic
- Build on what others say and find common ground
- Use a tone that's professional but approachable - like you're at a conference networking event

Remember: You're having real conversations with real people, not giving presentations or writing reports.
"""


def create_agent(
    name: str,
    description: str,
    system_prompt: str,
    model_name: str = "gpt-4.1",
) -> Agent:
    """
    Create an agent with proper documentation and configuration.

    Args:
        name: The agent's name
        description: Brief description of the agent's role
        system_prompt: The system prompt defining the agent's behavior
        model_name: The model to use for the agent

    Returns:
        Configured Agent instance
    """
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt + NATURAL_CONVERSATION_PROMPT,
        model_name=model_name,
        dynamic_temperature_enabled=True,
        output_type="str-all-except-first",
        streaming_on=False,
        max_loops=1,
    )


def create_demo_agents() -> List[Agent]:
    """
    Create a diverse set of AI agents with different specializations.

    Returns:
        List of configured Agent instances
    """
    agents = []

    # Quantitative Trading Agent
    agents.append(
        create_agent(
            name="QuantTrader-Alpha",
            description="Advanced quantitative trading and algorithmic analysis expert",
            system_prompt="""You are an expert quantitative trading agent with deep expertise in:
        - Algorithmic trading strategies and implementation
        - Statistical arbitrage and market making
        - Risk management and portfolio optimization
        - High-frequency trading systems
        - Market microstructure analysis
        - Quantitative research methodologies
        
        You focus on mathematical rigor, statistical significance, and risk-adjusted returns.
        You communicate in precise, technical terms while being collaborative with other experts.""",
        )
    )

    # Market Research Analyst Agent
    agents.append(
        create_agent(
            name="MarketAnalyst-Beta",
            description="Comprehensive market research and fundamental analysis specialist",
            system_prompt="""You are a senior market research analyst specializing in:
        - Fundamental analysis and company valuation
        - Economic indicators and macro trends
        - Industry sector analysis and comparative studies
        - ESG (Environmental, Social, Governance) factors
        - Market sentiment and behavioral finance
        - Long-term investment strategy
        
        You excel at identifying long-term trends, evaluating company fundamentals, and 
        providing strategic investment insights based on thorough research.""",
        )
    )

    # Risk Management Agent
    agents.append(
        create_agent(
            name="RiskManager-Gamma",
            description="Enterprise risk management and compliance expert",
            system_prompt="""You are a chief risk officer focused on:
        - Portfolio risk assessment and stress testing
        - Regulatory compliance and reporting
        - Credit risk and counterparty analysis
        - Market risk and volatility modeling
        - Operational risk management
        - Risk-adjusted performance measurement
        
        You prioritize capital preservation, regulatory adherence, and sustainable growth.
        You challenge aggressive strategies and ensure proper risk controls.""",
        )
    )

    # Cryptocurrency & DeFi Agent
    agents.append(
        create_agent(
            name="CryptoExpert-Delta",
            description="Cryptocurrency and decentralized finance specialist",
            system_prompt="""You are a blockchain and DeFi expert specializing in:
        - Cryptocurrency market analysis and tokenomics
        - DeFi protocols and yield farming strategies
        - Blockchain technology and smart contract analysis
        - NFT markets and digital asset valuation
        - Cross-chain bridges and layer 2 solutions
        - Regulatory developments in digital assets
        
        You stay current with the rapidly evolving crypto ecosystem and can explain
        complex DeFi concepts while assessing their risks and opportunities.""",
        )
    )

    # Economic Policy Agent
    agents.append(
        create_agent(
            name="PolicyEconomist-Epsilon",
            description="Macroeconomic policy and central banking expert",
            system_prompt="""You are a macroeconomic policy expert focusing on:
        - Central bank policies and monetary transmission
        - Fiscal policy impacts on markets
        - International trade and currency dynamics
        - Inflation dynamics and interest rate cycles
        - Economic growth models and indicators
        - Geopolitical risk assessment
        
        You analyze how government policies and global economic trends affect
        financial markets and investment strategies.""",
        )
    )

    # Behavioral Finance Agent
    agents.append(
        create_agent(
            name="BehaviorAnalyst-Zeta",
            description="Behavioral finance and market psychology expert",
            system_prompt="""You are a behavioral finance expert specializing in:
        - Market psychology and investor sentiment
        - Cognitive biases in investment decisions
        - Social trading and crowd behavior
        - Market anomalies and inefficiencies
        - Alternative data and sentiment analysis
        - Neurofinance and decision-making processes
        
        You understand how human psychology drives market movements and can identify
        opportunities created by behavioral biases and market sentiment.""",
        )
    )

    return agents


def main():
    """Main demo function that runs the agent map simulation."""
    print("🚀 Starting Agent Map Simulation Demo")
    print("=" * 60)

    try:
        # Create the simulation
        print("📍 Setting up simulation environment...")
        simulation = AgentMapSimulation(
            map_width=50.0,
            map_height=50.0,
            proximity_threshold=8.0,  # Agents will talk when within 8 units
            update_interval=2.0,  # Update every 2 seconds
        )

        # Create and add agents
        print("👥 Creating specialized financial agents...")
        agents = create_demo_agents()

        for agent in agents:
            # Add each agent at a random position
            simulation.add_agent(
                agent=agent,
                movement_speed=2.0,  # Moderate movement speed
                conversation_radius=8.0,  # Same as proximity threshold
            )
            time.sleep(0.5)  # Small delay for visual effect

        print(f"\n✅ Added {len(agents)} agents to the simulation")

        # Set up visualization
        print("📊 Setting up live visualization...")
        print(
            "💡 If you don't see a window, the simulation will still run with text updates"
        )
        simulation.setup_visualization(figsize=(14, 10))

        # Start the simulation
        print("🏃 Starting simulation...")
        simulation.start_simulation()

        print("\n" + "=" * 60)
        print("🎮 SIMULATION CONTROLS:")
        print("  - Agents will move randomly around the map")
        print(
            "  - When agents get close, they'll start natural conversations"
        )
        print(
            "  - Watch the visualization window for real-time updates"
        )
        print(
            "  - If no window appears, check the text updates below"
        )
        print("  - Press Ctrl+C to stop the simulation")
        print("=" * 60)

        # Start live visualization in a separate thread-like manner
        try:
            print("\n🎬 Attempting to start live visualization...")
            simulation.start_live_visualization(update_interval=3.0)
        except Exception as e:
            print(f"📊 Visualization not available: {str(e)}")
            print("📊 Continuing with text-only updates...")

        # Monitor simulation with regular updates
        print(
            "\n📊 Monitoring simulation (text updates every 5 seconds)..."
        )
        for i in range(60):  # Run for 60 iterations (about 5 minutes)
            time.sleep(5)
            simulation.print_status()

            # Try to manually update visualization if it exists
            if simulation.fig is not None:
                try:
                    simulation.update_visualization()
                except Exception:
                    pass  # Ignore visualization errors

            # Check if we have enough conversations to make it interesting
            state = simulation.get_simulation_state()
            if state["total_conversations"] >= 3:
                print(
                    f"\n🎉 Great! We've had {state['total_conversations']} conversations so far."
                )
                print("Continuing simulation...")

    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")

    except Exception as e:
        print(f"\n❌ Error in simulation: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up and save results
        print("\n🧹 Cleaning up simulation...")

        try:
            simulation.stop_simulation()

            # Save conversation summary
            print("💾 Saving conversation summary...")
            filename = simulation.save_conversation_summary()

            # Final status
            simulation.print_status()

            print("\n📋 Simulation complete!")
            print(
                f"📄 Detailed conversation log saved to: {filename}"
            )
            print(
                "🏁 Thank you for running the Agent Map Simulation demo!"
            )

        except Exception as e:
            print(f"⚠️  Error during cleanup: {str(e)}")
