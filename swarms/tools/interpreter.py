import interpreter

def compile(task: str):
    task = interpreter.chat(task)
    interpreter.chat()
    interpreter.reset()


