from ai_acceleerated_learning.Vocal import Vocal

vocal = Vocal()


def test_pass():
    assert (
        vocal.generate_video(
            "I love to play basketball, and I am a very good player.",
            "basketball",
        )
        == "Successfully generated a YouTube video for your prompt: I"
        " love to play basketball, and I am a very good player."
    )


def test_invalid_sports():
    assert (
        vocal.generate_video(
            "I just ate some delicious tacos", "tacos"
        )
        == "Invalid sports entered!! Please enter a valid sport."
    )


def test_invalid_prompt():
    assert (
        vocal.generate_video(987, "basketball")
        == "Invalid prompt entered!! Please enter a valid prompt."
    )


def test_not_string():
    assert (
        vocal.generate_video(789, 234)
        == "Invalid prompt and sports entered!! Please enter valid"
        " prompt and sport."
    )
