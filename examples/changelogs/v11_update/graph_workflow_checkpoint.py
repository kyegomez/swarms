from swarms import GraphWorkflow

wf = GraphWorkflow(
    checkpoint_dir="./checkpoints",  # new parameter
    checkpoint_interval=5,  # save every 5 steps
)
wf.add_node(...)
wf.add_edge(...)

# If interrupted, resume from the last checkpoint:
wf2 = GraphWorkflow.load_checkpoint(
    "./checkpoints/my_workflow_step_10.json"
)
wf2.run()  # continues from step 11
