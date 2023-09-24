def select_next_speaker(
        step: int,
        agents,
        director
) -> int:
    #if the step if even => director
    #=> director selects next speaker
    if step % 2 == 1:
        idx = 0
    else:
        idx = director.select_next_speaker() + 1
    return idx
