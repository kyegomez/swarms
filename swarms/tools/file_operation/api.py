from pathlib import Path
from ..tool import Tool


def build_tool(config) -> Tool:
    tool = Tool(
        "File Operation Tool",
        "Write / read file to / from disk",
        name_for_model="file_operation",
        description_for_model="Plugin for operating files",
        logo_url=None,
        contact_email=None,
        legal_info_url=None,
    )

    @tool.get("/write_file")
    def write_file(file_path: str, text: str) -> str:
        """write file to disk"""
        write_path = Path(file_path)
        try:
            write_path.parent.mkdir(exist_ok=True, parents=False)
            with write_path.open("w", encoding="utf-8") as f:
                f.write(text)
            return f"File written successfully to {file_path}."
        except Exception as e:
            return "Error: " + str(e)

    @tool.get("/read_file")
    def read_file(file_path: str) -> str:
        """read file from disk"""
        read_path = Path(file_path)
        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    return tool
