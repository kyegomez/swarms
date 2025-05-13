import pytest
from swarms.utils.xml_utils import dict_to_xml, to_xml_string
import xml.etree.ElementTree as ET

def test_dict_to_xml_simple():
    d = {"foo": "bar", "baz": 1}
    elem = dict_to_xml("root", d)
    xml_str = ET.tostring(elem, encoding="unicode")
    assert "<foo>bar</foo>" in xml_str
    assert "<baz>1</baz>" in xml_str

def test_dict_to_xml_nested():
    d = {"foo": {"bar": "baz"}}
    elem = dict_to_xml("root", d)
    xml_str = ET.tostring(elem, encoding="unicode")
    assert "<foo>" in xml_str and "<bar>baz</bar>" in xml_str

def test_dict_to_xml_list():
    d = {"items": [1, 2, 3]}
    elem = dict_to_xml("root", d)
    xml_str = ET.tostring(elem, encoding="unicode")
    assert xml_str.count("<item>") == 3
    assert "<item>1</item>" in xml_str

def test_to_xml_string_dict():
    d = {"foo": "bar"}
    xml = to_xml_string(d, root_tag="root")
    assert xml.startswith("<root>") and "<foo>bar</foo>" in xml

def test_to_xml_string_list():
    data = [{"a": 1}, {"b": 2}]
    xml = to_xml_string(data, root_tag="root")
    assert xml.startswith("<root>") and xml.count("<item>") == 2

def test_to_xml_string_scalar():
    xml = to_xml_string("hello", root_tag="root")
    assert xml == "<root>hello</root>"

def test_dict_to_xml_edge_cases():
    d = {"empty": [], "none": None, "bool": True}
    elem = dict_to_xml("root", d)
    xml_str = ET.tostring(elem, encoding="unicode")
    assert "<empty />" in xml_str or "<empty></empty>" in xml_str
    assert "<none>None</none>" in xml_str
    assert "<bool>True</bool>" in xml_str
