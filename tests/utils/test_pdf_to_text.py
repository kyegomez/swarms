import pytest
import pypdf
from swarms.utils import pdf_to_text


@pytest.fixture
def pdf_file(tmpdir):
    pdf_writer = pypdf.PdfWriter()
    pdf_page = pypdf.PageObject.create_blank_page(None, 200, 200)
    pdf_writer.add_page(pdf_page)
    pdf_file = tmpdir.join("temp.pdf")
    with open(pdf_file, "wb") as output:
        pdf_writer.write(output)
    return str(pdf_file)


def test_valid_pdf_to_text(pdf_file):
    result = pdf_to_text(pdf_file)
    assert isinstance(result, str)


def test_non_existing_file():
    with pytest.raises(FileNotFoundError):
        pdf_to_text("non_existing_file.pdf")


def test_passing_non_pdf_file(tmpdir):
    file = tmpdir.join("temp.txt")
    file.write("This is a test")
    with pytest.raises(
        Exception,
        match=r"An error occurred while reading the PDF file",
    ):
        pdf_to_text(str(file))


@pytest.mark.parametrize("invalid_pdf_file", [None, 123, {}, []])
def test_invalid_pdf_to_text(invalid_pdf_file):
    with pytest.raises(Exception):
        pdf_to_text(invalid_pdf_file)
