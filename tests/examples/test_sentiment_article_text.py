"""Offline regressions for the sentiment example's article extraction."""

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def extractor(monkeypatch: pytest.MonkeyPatch) -> type:
    """Load the real extractor with optional HTML parsing and no API calls."""
    pytest.importorskip("bs4")
    modules = {
        name: ModuleType(name)
        for name in ("swarms", "dotenv", "requests")
    }
    modules["swarms"].Agent = Mock()
    modules["dotenv"].load_dotenv = Mock()
    modules["requests"].get = Mock(
        side_effect=AssertionError("Unexpected HTTP request")
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/demos/finance/sentiment_news_analysis.py"
    )
    return runpy.run_path(str(path))["ArticleExtractor"]


@pytest.mark.parametrize(
    ("html", "domain"),
    [
        (
            '<div class="caas-body"><h2>Markets</h2><p>rose</p></div>',
            "finance.yahoo.com",
        ),
        (
            "<article><h2>Markets</h2><p>rose</p></article>",
            "example.com",
        ),
        ("<p>Markets<br>rose</p>", "example.com"),
        ("<div>Markets<br>rose</div>", "example.com"),
    ],
)
def test_text_boundaries_survive_each_extraction_path(
    extractor: type, html: str, domain: str
) -> None:
    """Headings, paragraphs and line breaks must not join adjacent words."""
    assert (
        extractor.extract_article_content(html, domain)
        == "Markets rose"
    )


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p>  Markets\n rose  </p>", "Markets rose"),
        (
            "<p>Profit was 1<b>2</b>.5 million.</p>",
            "Profit was 12.5 million.",
        ),
        (
            "<nav>Navigation</nav><article>Markets rose</article>",
            "Markets rose",
        ),
    ],
)
def test_whitespace_cleaning_and_navigation_removal_remain(
    extractor: type, html: str, expected: str
) -> None:
    """Keep whitespace normalization and exclusion of navigation text."""
    assert (
        extractor.extract_article_content(html, "example.com")
        == expected
    )
