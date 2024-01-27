from swarms.structs.agent import Agent


def agent_wrapper(ClassToWrap):
    """
    This function takes a class 'ClassToWrap' and returns a new class that
    inherits from both 'ClassToWrap' and 'Agent'. The new class overrides
    the '__init__' method of 'Agent' to call the '__init__' method of 'ClassToWrap'.

    Args:
        ClassToWrap (type): The class to be wrapped and made to inherit from 'Agent'.

    Returns:
        type: The new class that inherits from both 'ClassToWrap' and 'Agent'.
    """

    class WrappedClass(ClassToWrap, Agent):
        def __init__(self, *args, **kwargs):
            try:
                Agent.__init__(self, *args, **kwargs)
                ClassToWrap.__init__(self, *args, **kwargs)
            except Exception as e:
                print(f"Error initializing WrappedClass: {e}")
                raise e

    return WrappedClass
