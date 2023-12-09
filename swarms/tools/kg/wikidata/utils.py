from copy import deepcopy
from SPARQLWrapper import SPARQLWrapper, JSON
import csv
import regex as re
import os

DIRPATH = os.path.dirname(os.path.abspath(__file__))

# Dictionary to store all property labels and description


class PropertyDetails:
    def __init__(self):
        self.prop_details = dict()
        with open(f"{DIRPATH}/property.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            for prop in reader:
                self.prop_details[prop[0]] = [prop[1], prop[2]]

    def get_details(self, prop_id):
        return self.prop_details.get(prop_id, ["", ""])


sid_num = 0
propdetails = PropertyDetails()


def convert_sparql_to_backend(query):
    all_var_str = "[() ]\?[a-zA-Z0-9_-]+[() ]"

    filter_str = r"\(.+\(.* (\?.+)\) [Aa][Ss].*\)"

    sparql_split = query.split("\n")
    select = sparql_split[0]
    select += " "
    sel_list = re.findall(all_var_str, select, overlapped=True)
    sel_list = [sel[1:-1] for sel in sel_list]

    rm_list = re.findall(filter_str, select)

    for sel in rm_list:
        sel_list.remove(sel)

    # print(sel_list)

    added_sel_list = []
    basic_sel_list = []

    for sel in sel_list:
        if len(sel) > 0 and sel[0] == "?":
            basic_sel_list.append(sel)
            added_sel_list.append(sel + "Label")
            added_sel_list.append(sel + "Description")

    if len(rm_list) == 0:
        for sel in added_sel_list:
            select += " " + sel

    # print(select)

    sparql_split[0] = select

    service_pos = -1

    query = "\n".join(sparql_split)

    for i in range(len(query) - 1, -1, -1):
        if query[i] == "}":
            service_pos = i
            break

    query = (
        query[:service_pos]
        + 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }\n'
        + query[service_pos:]
        + "\nLIMIT 200"
    )
    basic_sel_list = [b[1:] for b in basic_sel_list]
    return query, basic_sel_list


def get_property_details_with_name(url: str, name: str):
    id = url.split("/")[-1]
    checkConst = url.split("/")[4]
    if len(id) > 0 and id[0] == "#":
        return {
            name: id,
            name + "Label": "",
            name + "Description": "",
            # name + 'uri': ''
        }
    elif checkConst[0] == '"':
        label = url.split('"')[1]
        type = url.split("<")[-1]
        type = type[0 : len(type) - 1]
        return {
            name: "",
            name + "Label": label,
            name + "Description": "",
            # name + 'uri': '',
            # 'type': type
        }
    prop = propdetails.get_details(id)
    id = id.split("+")
    if len(id) == 1:
        return {
            name: id[0],
            name + "Label": prop[0],
            name + "Description": prop[1],
            # 'propuri': url
        }
    labels = [propdetails.get_details(id_)[0] for id_ in id]

    return {
        name: "+".join(id),
        name + "Label": "+".join(labels),
        name + "Description": "",
        # name + 'uri': ''
    }


def convert(dictt):
    for key in dictt:
        dictt[key] = {"value": dictt[key]}
    return dictt


def getResult(query):
    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql",
        agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    )

    sparql.setQuery(query)

    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()

    return result["results"]["bindings"]


def get_details_sparql(id):
    condition = 'wd:{} rdfs:label ?label. wd:{} schema:description ?description. FILTER(lang(?label) = "en" && lang(?description) = "en")'.format(
        id, id
    )
    query = "SELECT ?label ?description \nWHERE\n{\n" + condition + "\n}\n LIMIT 1"

    return query


# Get label & description of an entity


def get_entity_details(url: str):
    id = url.split("/")[-1]

    if len(id) > 0 and id[0] == "#":
        return {"id": id, "label": None, "description": None, "uri": None}

    if not (len(id) > 0 and id[0] in ["P", "Q"]):
        return {"id": id, "label": None, "description": None, "uri": None}

    if url[0:4] != "http":
        return {"id": None, "label": None, "description": None, "uri": None}

    if url[7:23] != "www.wikidata.org":
        return {"id": None, "label": None, "description": None, "uri": url}

    result = getResult(get_details_sparql(id))

    if len(result) == 0:
        return {"id": id, "label": None, "description": None, "uri": url, "type": ""}

    response = {
        "id": id,
        "label": result[0].get("label", {"value": ""})["value"],
        "description": result[0].get("description", {"value": ""})["value"],
        "uri": url,
        "type": "uri",
    }

    return response


# Get label & description of a property
def get_property_details(url: str):
    id = url.split("/")[-1]
    checkConst = url.split("/")[4]
    if len(id) > 0 and id[0] == "#":
        return {
            "prop": id,
            "propLabel": "",
            "propDescription": "",
            # 'propuri': ''
        }
    elif checkConst[0] == '"':
        label = url.split('"')[1]
        type = url.split("<")[-1]
        type = type[0 : len(type) - 1]
        return {
            "prop": "",
            "propLabel": label,
            "propDescription": "",
            # 'propuri': '',
            # 'type': type
        }
    prop = propdetails.get_details(id)
    id = id.split("+")
    if len(id) == 1:
        return {
            "prop": id[0],
            "propLabel": prop[0],
            "propDescription": prop[1],
            # 'propuri': url
        }
    labels = [propdetails.get_details(id_)[0] for id_ in id]

    return {
        "prop": "+".join(id),
        "propLabel": "+".join(labels),
        "propDescription": "",
        # 'propuri': ''
    }


def enc(i):
    assert "attr" in i and i["id"] != None
    global sid_num
    if i["attr"] == "tmp":
        return "?" + "tmp" + str(i["id"]) + "_"
    if i["attr"] == "val":
        return str(i["id"])
    if i["attr"] == "sid":
        return "?" + "sid_" + str(i["id"])

    if len(i["id"].split("|")) > 1:
        Str = ""
        for Id in i["id"].split("|"):
            sid_num += 1
            Str += i["attr"] + ":" + Id
            Str += "|"
        return Str[:-1]

    if i["attr"] == "wdt":
        sid_num += 1
        return "p:{} ?sid_{}.\n?sid_{} ps:{}".format(
            str(i["id"]), sid_num, sid_num, str(i["id"])
        )
    return i["attr"] + ":" + str(i["id"])


class Slot2Sparql:
    class selection:
        def __init__(self):
            self.str0 = "SELECT "  # 搜索的目标字符串
            self.str1 = ""
            self.str2 = ""
            self.select = []  # select后内容
            self.select_sid = []  # 最新statementId
            self.new_select = []  # count max min select 的tmp id
            self.trip = []  # 下方的搜索字符串
            self.tmp = []  # 临时变量
            self.state = []
            self.tail = []  # 尾部

            self.find_tuple_match = {}

        def give_str(self):
            need_label = len(self.str1) == 0

            str = self.str0

            for s in self.select:
                cur_enc = enc(s)
                str += cur_enc
                if need_label:
                    str += " {}Label {}Description".format(cur_enc, cur_enc)
                if len(self.select) == 1:
                    str += self.str1
                str += self.str2
                str += " "

            for s in self.select_sid:
                str += enc(s)

            str += "\nWHERE\n{\n"
            for s in self.trip:
                str += s
                if str[-1] != "{":
                    str += "."
                str += "\n"

            if need_label:
                str += 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }\n'
            str += "}"
            for s in self.tail:
                str += "\n"
                str += s

            str += "\n"
            return str

        def set_select(self, sele):
            self.str0 = "SELECT "
            self.select = [sele]
            self.str1 = ""

        def clear_all(self):
            self.str0 = "SELECT "  # 搜索的目标字符串
            self.str1 = ""
            self.str2 = ""
            self.select = []  # select后内容
            self.trip = []  # 下方的搜索字符串
            self.tmp = []  # 临时变量
            self.state = []
            self.tail = []  # 尾部

        def getCount(self):
            str = self.str0

            str += "(COUNT("

            for s in self.select:
                cur_enc = enc(s)
                str += cur_enc
                if len(self.select) == 1:
                    str += self.str1
                str += " "

            str += ") AS ?cnt)"

            str += "\nWHERE\n{\n"
            for s in self.trip:
                str += s
                str += "."
                str += "\n"

            str += "}"
            for s in self.tail:
                str += "\n"
                str += s

            str += "\n"
            return str

    def __init__(self):
        self.select_lst = []
        self.num = 0
        sid_num = 0

    def clear_all(self):
        self.select_lst = []
        self.num = 0

    def new_select_lst(self):
        self.select_lst.append(self.selection())

    def prev_select_lst(self, idx):
        self.select_lst.append(deepcopy(self.select_lst[idx]))

    def find_select_lst(self, tar):
        assert tar["attr"] == "tmp" and tar["id"] < self.num
        if tar in self.select_lst[-1].tmp:
            return
        for i in range(len(self.select_lst) - 2, -1, -1):
            if tar in self.select_lst[i].select:
                self.select_lst[-1].trip += self.select_lst[i].trip  # 下方的搜索字符串
                self.select_lst[-1].state += self.select_lst[i].state
                self.select_lst[-1].tmp += self.select_lst[i].tmp
                self.select_lst[-1].tail += self.select_lst[i].tail  # 尾部
                return

    def head(
        self,
    ):
        pass

    def body(
        self,
    ):
        pass

    def find_tuple(self, tup):
        self.new_select_lst()
        target = []
        for i in tup:
            if tup[i]["attr"] == "tmp" and tup[i]["id"] == None:
                # 新的临时变量
                tup[i]["id"] = self.num
                self.num += 1
                target.append(tup[i])
                self.select_lst[-1].find_tuple_match[i] = enc(tup[i])[1:]
                self.select_lst[-1].tmp.append(tup[i])
            elif tup[i]["attr"] == "tmp":
                assert tup[i]["id"] < self.num
                self.find_select_lst(tup[i])
                target.append(tup[i])
                self.select_lst[-1].find_tuple_match[i] = enc(tup[i])[1:]

        if target == []:
            is_triplet_full = True
            for i in tup:
                if tup[i]["attr"] == "tmp":
                    self.find_select_lst(tup[i])
                    target.append(tup[i])
                    self.select_lst[-1].find_tuple_match[i] = enc(tup[i])[1:]
                    break
        else:
            is_triplet_full = False

        self.select_lst[-1].select = target
        self.select_lst[-1].state.append([tup["x"], tup["y"], tup["z"]])

        if type(tup["y"]["id"]) == str:
            y_id_splited = tup["y"]["id"].split("+")
        else:
            y_id_splited = []

        tmpXZ = [tup["x"]]
        for i in range(len(y_id_splited) - 1):
            tmpXZ.append({"attr": "tmp", "id": self.num})
            self.num += 1
        tmpXZ.append(tup["z"])

        idx = 0
        str1 = ""
        if len(y_id_splited) != 0:
            for tmpY in y_id_splited:
                newY = {"attr": "wdt", "id": tmpY}
                str1 += enc(tmpXZ[idx])
                str1 += " "
                str1 += enc(newY)
                str1 += " "
                str1 += enc(tmpXZ[idx + 1])
                str1 += ".\n"
                idx += 1
        else:
            str1 += enc(tup["x"])
            str1 += " "
            str1 += enc(tup["y"])
            str1 += " "
            str1 += enc(tup["z"])
            str1 += ".\n"
        str1 = str1[:-2]

        print(str1)

        self.select_lst[-1].select_sid = [{"attr": "sid", "id": sid_num}]
        self.select_lst[-1].trip.append(str1)

        if is_triplet_full:
            self.change_tmpidx(target[0])

    def find_entity(self, ent1):
        self.new_select_lst()
        self.select_lst[-1].str0 += "DISTINCT "
        self.select_lst[-1].select = [{}, {}, {}]
        innerSelect = [{}, {}]

        for t in self.select_lst[-1].select:
            t["attr"] = "tmp"
            t["id"] = self.num
            self.num += 1
            self.select_lst[-1].tmp.append(t)
        for t in innerSelect:
            t["attr"] = "tmp"
            t["id"] = self.num
            self.num += 1

        if ent1["attr"] == "tmp":
            self.find_select_lst(ent1)
        # ent1位于三元组的头
        self.select_lst[-1].state.append(
            [
                ent1,
                self.select_lst[-1].select[0],
                self.select_lst[-1].select[1],
                self.select_lst[-1].select[2],
            ]
        )
        str1 = enc(ent1)
        str1 += " "
        str1 += enc(self.select_lst[-1].select[0])
        str1 += " "
        str1 += enc(self.select_lst[-1].select[1])
        self.select_lst[-1].trip.append("{")
        self.select_lst[-1].trip.append(str1)
        self.select_lst[-1].trip.append("}\nUNION\n{")
        str1 = enc(ent1)
        str1 += " "
        str1 += enc(innerSelect[0])
        str1 += " "
        str1 += enc(innerSelect[1])
        self.select_lst[-1].trip.append(str1)
        str1 = enc(innerSelect[1])
        str1 += " pq:P585 "
        str1 += enc(self.select_lst[-1].select[2])
        str1 += ";\n"
        str1 += enc(self.select_lst[-1].select[0])
        str1 += " "
        str1 += enc(self.select_lst[-1].select[1])
        self.select_lst[-1].trip.append(str1)
        self.select_lst[-1].trip.append("}")

        if ent1["attr"] == "wds":
            str1 = 'FILTER(STRSTARTS ( STR ( {} ), "http://www.wikidata.org/prop/" ))'.format(
                enc(self.select_lst[-1].select[0])
            )
        else:
            str1 = 'FILTER(STRSTARTS ( STR ( {} ), "http://www.wikidata.org/prop/direct/" ) ||  STRSTARTS ( STR ( {} ),"http://www.wikidata.org/prop/statement/" ))'.format(
                enc(self.select_lst[-1].select[0]), enc(self.select_lst[-1].select[0])
            )
        self.select_lst[-1].trip.append(str1)

    def find_entity_by_tail(self, ent1):
        self.new_select_lst()
        self.select_lst[-1].str0 += "DISTINCT "
        self.select_lst[-1].select = [{}, {}]
        for t in self.select_lst[-1].select:
            t["attr"] = "tmp"
            t["id"] = self.num
            self.num += 1
            self.select_lst[-1].tmp.append(t)
        if ent1["attr"] == "tmp":
            self.find_select_lst(ent1)
        # ent1位于三元组的尾
        self.select_lst[-1].state.append(
            [self.select_lst[-1].select[0], self.select_lst[-1].select[1], ent1]
        )
        str1 = enc(self.select_lst[-1].select[0])
        str1 += " "
        str1 += enc(self.select_lst[-1].select[1])
        str1 += " "
        str1 += enc(ent1)
        self.select_lst[-1].trip.append(str1)

        str1 = 'FILTER(STRSTARTS ( STR ( {} ), "http://www.wikidata.org/entity/Q" ))'.format(
            enc(self.select_lst[-1].select[0])
        )
        self.select_lst[-1].trip.append(str1)

        str1 = (
            'FILTER(STRSTARTS ( STR ( {} ), "http://www.wikidata.org/prop/" ))'.format(
                enc(self.select_lst[-1].select[1])
            )
        )
        self.select_lst[-1].trip.append(str1)

    def find_entity_by_relation(self, ent1):
        self.new_select_lst()
        self.select_lst[-1].str0 += "DISTINCT "
        self.select_lst[-1].select = [{}, {}]
        for t in self.select_lst[-1].select:
            t["attr"] = "tmp"
            t["id"] = self.num
            self.num += 1
            self.select_lst[-1].tmp.append(t)
        if ent1["attr"] == "tmp":
            self.find_select_lst(ent1)
        # ent1位于三元组的尾
        self.select_lst[-1].state.append(
            [self.select_lst[-1].select[0], self.select_lst[-1].select[1], ent1]
        )
        str1 = enc(self.select_lst[-1].select[0])
        str1 += " "
        str1 += enc(ent1)
        str1 += " "
        str1 += enc(self.select_lst[-1].select[1])
        self.select_lst[-1].trip.append(str1)

        str1 = 'FILTER(STRSTARTS ( STR ( {} ), "http://www.wikidata.org/entity/Q" ))'.format(
            enc(self.select_lst[-1].select[0])
        )
        self.select_lst[-1].trip.append(str1)

    def binary_operation(self, ent1, op, ent2):
        if op in [">", "<", "=", "!=", ">=", "<="]:
            self.new_select_lst()
            assert ent1["attr"] == "tmp"
            self.find_select_lst(ent1)
            # 使用 filter 表示比较关系
            str1 = "FILTER ("
            str1 += enc(ent1)
            str1 += " "
            str1 += op
            str1 += " "
            str1 += enc(ent2)
            str1 += ")"
            self.select_lst[-1].trip.append(str1)
            self.select_lst[-1].select = [ent1]
            self.change_tmpidx(ent1)
            if ent2["attr"] == "tmp":
                self.select_lst[-1].select.append(ent2)

        elif op in ["+", "-", "*", "/"]:
            self.new_select_lst()
            if ent1["attr"] == "tmp":
                self.find_select_lst(ent1)
            if ent2["attr"] == "tmp":
                self.find_select_lst(ent2)
            # 使用新的临时变量
            # BIND(?tmpxx / 365.2425 AS ?tmpxx).
            t = {}
            t["attr"] = "tmp"
            t["id"] = self.num
            self.num += 1
            self.select_lst[-1].select = [t]
            self.select_lst[-1].tmp.append(t)

            str1 = "BIND ("
            str1 += enc(ent1)
            str1 += " "
            str1 += op
            str1 += " "
            str1 += enc(ent2)
            str1 += " AS "
            str1 += enc(t)
            str1 += ")."
            self.select_lst[-1].trip.append(str1)

        elif op in ["&&", "||", "~"]:
            self.new_select_lst()
            assert ent1["attr"] == ent2["attr"] == "tmp"
            self.select_lst[-1].trip.append("{")
            self.find_select_lst(ent1)
            if op == "&&":
                pass
            elif op == "||":
                self.select_lst[-1].trip.append("}\nUNION\n{")
            else:
                self.select_lst[-1].trip.append("}\nMINUS\n{")
            self.find_select_lst(ent2)
            self.select_lst[-1].trip.append("}")
            # 使用新的临时变量
            # BIND(?tmpxx / 365.2425 AS ?tmpxx).
            t = {}
            t["attr"] = "tmp"
            t["id"] = self.num
            self.num += 1
            tmp = []
            self.select_lst[-1].select = [t]
            self.select_lst[-1].tmp.append(t)
            self.select_lst[-1].tmp.remove(ent1)
            self.select_lst[-1].tmp.remove(ent2)
            for line in self.select_lst[-1].trip:
                nline = line.replace(enc(ent1), enc(t))
                nline = nline.replace(enc(ent2), enc(t))
                tmp.append(nline)
            self.select_lst[-1].trip = tmp
            for line in self.select_lst[-1].state:
                for i in line:
                    if i == ent1 or i == ent2:
                        i = t
            tmp = []
            for line in self.select_lst[-1].tail:
                nline = line.replace(enc(ent1), enc(t))
                nline = nline.replace(enc(ent2), enc(t))
                tmp.append(nline)
            self.select_lst[-1].tail = tmp

    def unitary_operation(self, ent, op, last_sparql_idx):
        if op in ["ORDER", "GROUP (ASC)", "GROUP (DESC)"]:
            self.new_select_lst()
            self.find_select_lst(ent)
            self.select_lst[-1].select = [ent]
            str1 = op.split(" ")[0] + " BY "
            str1 += enc(ent)
            if "GROUP" in op.split(" "):
                str1 += " {}Label {}Description".format(enc(ent), enc(ent))
                if op.split(" ")[-1] == "(DESC)":
                    str1 += "\nORDER BY DESC(?cnt)"
                else:
                    str1 += "\nORDER BY ?cnt"

            self.select_lst[-1].tail.append(str1)
            self.change_tmpidx(ent)

            if "GROUP" in op.split(" "):
                self.select_lst[-1].str2 = " (COUNT({}) AS ?cnt) ".format(
                    enc(self.select_lst[-1].select[0])
                )

        elif op in ["LIMIT", "OFFSET"]:
            self.prev_select_lst(last_sparql_idx)
            str1 = op + " "
            str1 += enc(ent)
            self.select_lst[-1].tail.append(str1)
            self.change_tmpidx(self.select_lst[-1].select[0])
            self.select_lst[-1].new_select = self.select_lst[-1].select

        elif op in ["DISTINCT", "REDUCED"]:
            self.new_select_lst()
            self.find_select_lst(ent)
            self.select_lst[-1].select = [ent]
            self.select_lst[-1].str0 += op
            self.select_lst[-1].str0 += " "

        elif op in ["MIN", "MAX", "AVG", "SUM", "COUNT", "SAMPLE"]:
            self.new_select_lst()
            self.find_select_lst(ent)

            t = {}
            t["attr"] = "tmp"
            t["id"] = self.num
            self.num += 1
            self.select_lst[-1].new_select = [t]
            self.select_lst[-1].tmp.append(t)

            self.select_lst[-1].select = [ent]

            self.select_lst[-1].str0 += "("
            self.select_lst[-1].str0 += op
            self.select_lst[-1].str0 += "("
            self.select_lst[-1].str1 += ") AS "
            self.select_lst[-1].str1 += enc(t)
            self.select_lst[-1].str1 += ")"

    def give_str(self, sparqlIdx=-1):
        return self.select_lst[sparqlIdx].give_str(), self.select_lst[sparqlIdx].select

    def give_tmp(self, sparqlIdx=-1):
        return self.select_lst[sparqlIdx].tmp

    def change_tmpidx(self, ent1, sparqlIdx=-1):
        # 将ent1的tmp_id更新
        t = {}
        t["attr"] = "tmp"
        t["id"] = self.num
        self.num += 1
        tmp = []
        self.select_lst[sparqlIdx].select = [t]
        self.select_lst[sparqlIdx].tmp.append(t)
        self.select_lst[sparqlIdx].tmp.remove(ent1)
        for line in self.select_lst[sparqlIdx].trip:
            nline = line.replace(enc(ent1), enc(t))
            tmp.append(nline)
        self.select_lst[sparqlIdx].trip = tmp
        for line in self.select_lst[sparqlIdx].state:
            for i in line:
                if i == ent1:
                    i = t
        tmp = []
        for line in self.select_lst[sparqlIdx].tail:
            nline = line.replace(enc(ent1), enc(t))
            tmp.append(nline)
        self.select_lst[sparqlIdx].tail = tmp

        self.select_lst[sparqlIdx].str2 = self.select_lst[sparqlIdx].str2.replace(
            enc(ent1), enc(t)
        )
