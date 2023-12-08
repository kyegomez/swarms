import requests
from pydantic import BaseModel
from bs4 import BeautifulSoup
import json, random
from ...tool import Tool
from typing import List, Optional, Union


class ChemicalPropAPI:
    def __init__(self) -> None:
        self._endpoint = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"

    def get_name_by_cid(self, cid: str, top_k: Optional[int] = None) -> List[str]:
        html_doc = requests.get(f"{self._endpoint}cid/{cid}/synonyms/XML").text
        soup = BeautifulSoup(html_doc, "html.parser", from_encoding="utf-8")
        syns = soup.find_all("synonym")
        ans = []
        if top_k is None:
            top_k = len(syns)
        for syn in syns[:top_k]:
            ans.append(syn.text)
        return ans

    def get_cid_by_struct(self, smiles: str) -> List[str]:
        html_doc = requests.get(f"{self._endpoint}smiles/{smiles}/cids/XML").text
        soup = BeautifulSoup(html_doc, "html.parser", from_encoding="utf-8")
        cids = soup.find_all("cid")
        if cids is None:
            return []
        ans = []
        for cid in cids:
            ans.append(cid.text)
        return ans

    def get_cid_by_name(self, name: str, name_type: Optional[str] = None) -> List[str]:
        url = f"{self._endpoint}name/{name}/cids/XML"
        if name_type is not None:
            url += f"?name_type={name_type}"
        html_doc = requests.get(url).text
        soup = BeautifulSoup(html_doc, "html.parser", from_encoding="utf-8")
        cids = soup.find_all("cid")
        if cids is None:
            return []
        ans = []
        for cid in cids:
            ans.append(cid.text)
        return ans

    def get_prop_by_cid(self, cid: str) -> str:
        html_doc = requests.get(
            f"{self._endpoint}cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,IUPACName,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,CovalentUnitCount/json"
        ).text
        return json.loads(html_doc)["PropertyTable"]["Properties"][0]


class GetNameResponse(BaseModel):

    """name list"""

    names: List[str]


class GetStructureResponse(BaseModel):

    """structure list"""

    state: int
    content: Optional[str] = None


class GetIDResponse(BaseModel):
    state: int
    content: Union[str, List[str]]


def build_tool(config) -> Tool:
    tool = Tool(
        "Chemical Property Plugin",
        description="looking up a chemical's property",
        name_for_model="Chemical Property",
        description_for_model="Plugin for looking up a chemical's property using a chemical knowledge base. All input should be a json like {'input': 'some input'}. Please use the provided questions and search step by step.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    if "debug" in config and config["debug"]:
        chemical_prop_api = config["chemical_prop_api"]
    else:
        chemical_prop_api = ChemicalPropAPI()

    @tool.get("/get_name")
    def get_name(cid: str):
        """prints the possible 3 synonyms of the queried compound ID"""
        ans = chemical_prop_api.get_name_by_cid(cid, top_k=3)
        return {"names": ans}

    @tool.get("/get_allname")
    def get_allname(cid: str):
        """prints all the possible synonyms (might be too many, use this function carefully)."""
        ans = chemical_prop_api.get_name_by_cid(cid)
        return {"names": ans}

    @tool.get("/get_id_by_struct")
    def get_id_by_struct(smiles: str):
        """prints the ID of the queried compound SMILES. This should only be used if smiles is provided or retrieved in the previous step. The input should not be a string, but a SMILES formula."""
        cids = chemical_prop_api.get_cid_by_struct(smiles)
        if len(cids) == 0:
            return {"state": "no result"}
        else:
            return {"state": "matched", "content": cids[0]}

    @tool.get("/get_id")
    def get_id(name: str):
        """prints the ID of the queried compound name, and prints the possible 5 names if the queried name can not been precisely matched,"""
        cids = chemical_prop_api.get_cid_by_name(name)
        if len(cids) > 0:
            return {"state": "precise", "content": cids[0]}

        cids = chemical_prop_api.get_cid_by_name(name, name_type="word")
        if len(cids) > 0:
            if name in get_name(cids[0]):
                return {"state": "precise", "content": cids[0]}

        ans = []
        random.shuffle(cids)
        for cid in cids[:5]:
            nms = get_name(cid)
            ans.append(nms)
        return {"state": "not precise", "content": ans}

    @tool.get("/get_prop")
    def get_prop(cid: str):
        """prints the properties of the queried compound ID"""
        return chemical_prop_api.get_prop_by_cid(cid)

    return tool
