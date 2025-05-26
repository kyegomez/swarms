import xml.etree.ElementTree as ET
from typing import Any


def dict_to_xml(tag: str, d: dict) -> ET.Element:
    """Convert a dictionary to an XML Element."""
    elem = ET.Element(tag)
    for key, val in d.items():
        child = ET.Element(str(key))
        if isinstance(val, dict):
            child.append(dict_to_xml(str(key), val))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    child.append(dict_to_xml(str(key), item))
                else:
                    item_elem = ET.Element("item")
                    item_elem.text = str(item)
                    child.append(item_elem)
        else:
            child.text = str(val)
        elem.append(child)
    return elem


def to_xml_string(data: Any, root_tag: str = "root") -> str:
    """Convert a dict or list to an XML string."""
    if isinstance(data, dict):
        elem = dict_to_xml(root_tag, data)
    elif isinstance(data, list):
        elem = ET.Element(root_tag)
        for item in data:
            if isinstance(item, dict):
                elem.append(dict_to_xml("item", item))
            else:
                item_elem = ET.Element("item")
                item_elem.text = str(item)
                elem.append(item_elem)
    else:
        elem = ET.Element(root_tag)
        elem.text = str(data)
    return ET.tostring(elem, encoding="unicode")
