import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

EXAMPLE_DIR = (
    Path(__file__).resolve().parents[2] / "examples/multi_agent"
)
SIMULATION_DIR = EXAMPLE_DIR / "simulations/euroswarm_parliament"
EXAMPLE = SIMULATION_DIR / "wikipedia_personality_scraper.py"
RECORD = (
    "<mep><fullName>{name}</fullName><country> Example </country>"
    "<politicalGroup>Group</politicalGroup><id>{id}</id>"
    "{party}</mep>"
)
PARTY = "<nationalPoliticalGroup>{}</nationalPoliticalGroup>"


class TestWikipediaMepXml(unittest.TestCase):
    def parse(self, xml: str) -> tuple:
        logger = Mock()
        with (
            patch.dict(
                sys.modules,
                {
                    "requests": SimpleNamespace(),
                    "loguru": SimpleNamespace(logger=logger),
                },
            ),
            patch(
                "socket.socket",
                side_effect=AssertionError("No network"),
            ),
            patch(
                "socket.create_connection",
                side_effect=AssertionError("No network"),
            ),
            TemporaryDirectory() as directory,
        ):
            namespace = runpy.run_path(str(EXAMPLE))
            scraper = object.__new__(
                namespace["WikipediaPersonalityScraper"]
            )
            scraper.verbose = False
            path = Path(directory) / "EU.xml"
            path.write_text(xml, encoding="utf-8")
            results = scraper.extract_mep_data_from_xml(str(path))
        return results, logger

    def test_entities_are_decoded(self) -> None:
        results, _ = self.parse(
            RECORD.format(
                name="First &amp; Second",
                id="1",
                party=PARTY.format("Party &amp; Friends"),
            )
        )
        self.assertEqual(results[0]["full_name"], "First & Second")
        self.assertEqual(
            results[0]["national_party"], "Party & Friends"
        )

    def test_empty_party_does_not_merge_adjacent_members(
        self,
    ) -> None:
        results, _ = self.parse(
            "<meps>"
            + RECORD.format(
                name="First",
                id="1",
                party="<nationalPoliticalGroup/>",
            )
            + RECORD.format(
                name="Second",
                id="2",
                party=PARTY.format("Party"),
            )
            + "</meps>"
        )
        self.assertEqual(
            [row["mep_id"] for row in results], ["1", "2"]
        )
        parties = [row["national_party"] for row in results]
        self.assertEqual(parties, ["", "Party"])

    def test_field_order_does_not_drop_member(self) -> None:
        results, _ = self.parse(
            "<meps><mep><id>1</id><country>Example</country>"
            "<nationalPoliticalGroup>Party</nationalPoliticalGroup>"
            "<fullName>First</fullName><politicalGroup>Group</politicalGroup>"
            "</mep></meps>"
        )
        self.assertEqual(
            results,
            [
                {
                    "full_name": "First",
                    "country": "Example",
                    "political_group": "Group",
                    "mep_id": "1",
                    "national_party": "Party",
                }
            ],
        )

    def test_standard_record_keeps_keys_and_strips_whitespace(
        self,
    ) -> None:
        results, _ = self.parse(
            RECORD.format(
                name=" First ",
                id=" 1 ",
                party=PARTY.format(" Party "),
            )
        )
        self.assertEqual(results[0]["full_name"], "First")
        self.assertEqual(results[0]["country"], "Example")
        self.assertEqual(results[0]["mep_id"], "1")
        self.assertEqual(results[0]["national_party"], "Party")

    def test_empty_document_returns_no_members(self) -> None:
        results, logger = self.parse("<meps/>")
        self.assertEqual(results, [])
        logger.error.assert_not_called()

    def test_malformed_xml_uses_existing_error_handler(self) -> None:
        results, logger = self.parse("<meps><mep>")
        self.assertEqual(results, [])
        logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
