import xml.etree.ElementTree as ET
from typing import Any


def dict_to_xml(tag: str, d: dict) -> ET.Element:
    """
    Convert a dictionary to an XML Element.

    Args:
        tag (str): The tag name for the root element
        d (dict): The dictionary to convert to XML

    Returns:
        ET.Element: An XML Element representing the dictionary structure

    Example:
        >>> data = {"person": {"name": "John", "age": 30}}
        >>> elem = dict_to_xml("root", data)
        >>> ET.tostring(elem, encoding="unicode")
        '<root><person><name>John</name><age>30</age></person></root>'
    """
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
    """
    Convert a dict or list to an XML string.

    Args:
        data (Any): The data to convert to XML. Can be a dictionary, list, or other value
        root_tag (str, optional): The tag name for the root element. Defaults to "root"

    Returns:
        str: An XML string representation of the input data

    Example:
        >>> data = {"person": {"name": "John", "age": 30}}
        >>> xml_str = to_xml_string(data)
        >>> print(xml_str)
        <root><person><name>John</name><age>30</age></person></root>

        >>> data = [1, 2, 3]
        >>> xml_str = to_xml_string(data)
        >>> print(xml_str)
        <root><item>1</item><item>2</item><item>3</item></root>
    """
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
