import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

EXAMPLE_DIR = (
    Path(__file__).resolve().parents[2] / "examples/guides/demos"
)
EXAMPLE = EXAMPLE_DIR / "real_estate/morgtate_swarm.py"


class TestMortgagePdfPagination(unittest.TestCase):
    def render_lines(
        self, content: str
    ) -> list[tuple[int, float, str]]:
        canvas = Mock(spec=["drawString", "showPage", "save"])
        agent = Mock()
        imports = {
            "swarms": SimpleNamespace(Agent=agent),
            "PyPDF2": ModuleType("PyPDF2"),
            "pydantic": SimpleNamespace(
                BaseModel=object, Field=Mock()
            ),
            "reportlab.lib.pagesizes": SimpleNamespace(
                LETTER=(612, 792)
            ),
            "reportlab.pdfgen": SimpleNamespace(
                canvas=SimpleNamespace(
                    Canvas=Mock(return_value=canvas)
                )
            ),
        }
        with (
            patch.dict(sys.modules, imports),
            patch(
                "socket.socket",
                side_effect=AssertionError("No network"),
            ),
            patch(
                "socket.create_connection",
                side_effect=AssertionError("No network"),
            ),
        ):
            namespace = runpy.run_path(str(EXAMPLE))
            swarm = object.__new__(
                namespace["MortgageUnderwritingSwarm"]
            )
            swarm.generate_pdf_report(content, "unused.pdf")
        agent.assert_not_called()
        canvas.save.assert_called_once_with()
        page = 1
        lines = []
        for call in canvas.method_calls:
            if call[0] == "showPage":
                page += 1
            elif call[0] == "drawString":
                x, y, text = call.args
                self.assertEqual(x, 50)
                self.assertGreater(y, 50)
                self.assertLessEqual(y, 742)
                lines.append((page, y, text))
        return lines

    def test_long_paragraph_spans_pages_without_clipping(
        self,
    ) -> None:
        content = "x" * 6000
        lines = self.render_lines(content)
        self.assertEqual(len(lines), 67)
        self.assertEqual(lines[-1][0], 2)
        self.assertEqual("".join(line[2] for line in lines), content)

    def test_wrapped_paragraph_near_bottom_starts_next_page(
        self,
    ) -> None:
        content = "Header\n" * 46 + "x" * 270
        lines = self.render_lines(content)
        self.assertEqual(lines[46], (1, 52, "x" * 90))
        self.assertEqual(lines[47], (2, 742, "x" * 90))
        self.assertEqual(lines[48], (2, 727, "x" * 90))

    def test_separate_lines_still_paginate(self) -> None:
        lines = self.render_lines("\n".join(["Line"] * 100))
        self.assertEqual(len(lines), 100)
        self.assertEqual(lines[-1][0], 3)
        self.assertTrue(all(line[2] == "Line" for line in lines))

    def test_short_report_preserves_blank_lines(self) -> None:
        lines = self.render_lines("## Heading\n\nParagraph.")
        self.assertEqual(
            lines,
            [
                (1, 742, "Heading"),
                (1, 727, ""),
                (1, 712, "Paragraph."),
            ],
        )

    def test_exact_wrap_width_has_no_extra_blank_line(self) -> None:
        self.assertEqual(
            self.render_lines("x" * 180),
            [(1, 742, "x" * 90), (1, 727, "x" * 90)],
        )

    def test_empty_report_retains_blank_line(self) -> None:
        self.assertEqual(self.render_lines(""), [(1, 742, "")])


if __name__ == "__main__":
    unittest.main()
