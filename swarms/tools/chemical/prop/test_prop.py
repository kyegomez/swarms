from fastapi.testclient import TestClient
from .api import build_tool, ChemicalPropAPI
from typing import Tuple, Optional, List


class ChemicalPropMock(ChemicalPropAPI):
    def __init__(self) -> None:
        self._endpoint = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"

    def get_name_by_cid(self, cid: str, top_k: Optional[int] = None) -> List[str]:
        ans = ["A", "B", "C", "D", "E"]
        if top_k is None:
            top_k = len(ans)
        return ans[:top_k]

    def get_cid_by_struct(self, smiles: str) -> List[str]:
        return ["123"]

    def get_cid_by_name(self, name: str, name_type: Optional[str] = None) -> List[str]:
        return ["123"]

    def get_prop_by_cid(self, cid: str) -> str:
        return {"works": "well"}


app = build_tool({"debug": True, "chemical_prop_api": ChemicalPropMock()})
client = TestClient(app)


def test_get_name():
    response = client.get("/get_name", params={"cid": 123})
    assert response.status_code == 200
    assert response.json() == {"names": ["A", "B", "C"]}


def test_get_all_names():
    response = client.get("/get_allname", params={"cid": 123})
    assert response.status_code == 200
    assert response.json() == {"names": ["A", "B", "C", "D", "E"]}


def test_get_id_by_struct():
    response = client.get("/get_id_by_struct", params={"smiles": "C1=CC=CC=C1"})
    assert response.status_code == 200
    assert response.json() == {"state": "matched", "content": "123"}


def test_get_id():
    response = client.get("/get_id", params={"name": "benzene"})
    assert response.status_code == 200
    assert response.json() == {
        "state": "precise",
        "content": "123",
    }


def test_get_prop():
    response = client.get("/get_prop", params={"cid": "123"})
    assert response.status_code == 200
    assert response.json() == {"works": "well"}
