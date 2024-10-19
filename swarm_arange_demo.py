from swarms.structs.swarm_arange import SwarmRearrange
from blackstone_pe.rearrange_example_blackstone import (
    blackstone_acquisition_analysis,
    blackstone_investment_strategy,
    blackstone_market_analysis,
)

swarm_arrange = SwarmRearrange(
    swarms=[
        blackstone_acquisition_analysis,
        blackstone_investment_strategy,
        blackstone_market_analysis,
    ],
    flow=f"{blackstone_acquisition_analysis.name} -> {blackstone_investment_strategy.name} -> {blackstone_market_analysis.name}, {blackstone_acquisition_analysis.name}",
)

print(
    swarm_arrange.run(
        "Analyze swarms, 150k revenue with 45m+ agents build, with 1.4m downloads since march 2024"
    )
)
