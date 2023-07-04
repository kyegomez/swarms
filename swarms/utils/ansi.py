class Code:
    def __init__(self, value: int):
        self.value = value

    def __str__(self):
        return "%d" % self.value


class Color(Code):
    def bg(self) -> "Color":
        self.value += 10
        return self

    def bright(self) -> "Color":
        self.value += 60
        return self

    @staticmethod
    def black() -> "Color":
        return Color(30)

    @staticmethod
    def red() -> "Color":
        return Color(31)

    @staticmethod
    def green() -> "Color":
        return Color(32)

    @staticmethod
    def yellow() -> "Color":
        return Color(33)

    @staticmethod
    def blue() -> "Color":
        return Color(34)

    @staticmethod
    def magenta() -> "Color":
        return Color(35)

    @staticmethod
    def cyan() -> "Color":
        return Color(36)

    @staticmethod
    def white() -> "Color":
        return Color(37)

    @staticmethod
    def default() -> "Color":
        return Color(39)


class Style(Code):
    @staticmethod
    def reset() -> "Style":
        return Style(0)

    @staticmethod
    def bold() -> "Style":
        return Style(1)

    @staticmethod
    def dim() -> "Style":
        return Style(2)

    @staticmethod
    def italic() -> "Style":
        return Style(3)

    @staticmethod
    def underline() -> "Style":
        return Style(4)

    @staticmethod
    def blink() -> "Style":
        return Style(5)

    @staticmethod
    def reverse() -> "Style":
        return Style(7)

    @staticmethod
    def conceal() -> "Style":
        return Style(8)


class ANSI:
    ESCAPE = "\x1b["
    CLOSE = "m"

    def __init__(self, text: str):
        self.text = text
        self.args = []

    def join(self) -> str:
        return ANSI.ESCAPE + ";".join([str(a) for a in self.args]) + ANSI.CLOSE

    def wrap(self, text: str) -> str:
        return self.join() + text + ANSI(Style.reset()).join()

    def to(self, *args: str):
        self.args = list(args)
        return self.wrap(self.text)


def dim_multiline(message: str) -> str:
    lines = message.split("\n")
    if len(lines) <= 1:
        return lines[0]
    return lines[0] + ANSI("\n... ".join([""] + lines[1:])).to(Color.black().bright())