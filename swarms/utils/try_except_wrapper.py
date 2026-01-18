def try_except_wrapper(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return None
