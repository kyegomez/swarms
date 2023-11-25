"""
Swarm of multi modal autonomous agents for manufacturing!
---------------------------------------------------------    
Health Security agent: Agent that monitors the health of working conditions: input image of factory output: health safety index 0.0 - 1.0 being the highest
Quality Control agent: Agent that monitors the quality of the product: input image of product output: quality index 0.0 - 1.0 being the highest
Productivity agent: Agent that monitors the productivity of the factory: input image of factory output: productivity index 0.0 - 1.0 being the highest
Safety agent: Agent that monitors the safety of the factory: input image of factory output: safety index 0.0 - 1.0 being the highest
Security agent: Agent that monitors the security of the factory: input image of factory output: security index 0.0 - 1.0 being the highest
Sustainability agent: Agent that monitors the sustainability of the factory: input image of factory output: sustainability index 0.0 - 1.0 being the highest
Efficiency agent: Agent that monitors the efficiency of the factory: input image of factory output: efficiency index 0.0 - 1.0 being the highest    


Flow:
health security agent -> quality control agent -> productivity agent -> safety agent -> security agent -> sustainability agent -> efficiency agent 
"""
