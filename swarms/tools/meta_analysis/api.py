import requests
from pydantic import BaseModel
import numpy as np
from bs4 import BeautifulSoup
import json
from ..tool import Tool
from typing import List, Optional, Union
from tenacity import retry, wait_random_exponential, stop_after_attempt
from numpy import dot
from numpy.linalg import norm
import openai
import pickle


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_response(prompt, token=2500):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.1,
        max_tokens=min(2500, token),
    )
    return response["choices"][0]["text"]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


standard = [
    "type of study",
    "purpose",
    "challenge",
    "dataset",
    "task",
    "design",
    "method",
    "backbone",
    "baseline",
    "model",
    "participant",
    "patient",
    "intervention",
    "group",
    "duration",
    "measurement",
    "performance",
    "result",
    "conclusion",
    "safety",
]
stan_emb = []
for item in standard:
    stan_emb.append(np.array(get_embedding(item)))


def dense(topic, docs):
    try:
        top_emb = np.array(get_embedding(topic))
    except Exception as e:
        return [-1]
    nm = norm(top_emb)
    doc_sim = []
    for ky in docs.keys():
        try:
            if docs[ky] is None:
                tmp = [0] * len(top_emb)
            elif len(docs[ky].split(" ")) > 100:
                tmp = get_embedding(docs[ky][0])
            else:
                tmp = [0] * len(top_emb)
        except Exception as e:
            tmp = [0] * len(top_emb)
        sim = dot(top_emb, np.array(tmp)) / (nm * norm(np.array(tmp)) + 1e-5)
        doc_sim.append(sim)
    doc_sim = np.array(doc_sim)
    return doc_sim


def semantic_query(term, retmax=100):
    if retmax > 100:
        print("semantic searching fewer than 100!")
        retmax = 100
    for cnt in range(10):
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search?query="
            + term.replace(" ", "+")
            + "&limit="
            + str(retmax)
            + "&fields=title,authors,abstract"
        )
        re = requests.get(url)
        results = json.loads(re.text)
        docs = {}
        if "data" in results:
            break
        tmp = 0
        while tmp < 10000:
            tmp += 1
    if "data" not in results:
        return docs
    for item in results["data"]:
        if item["abstract"] is not None:
            docs[item["title"]] = item["abstract"]
    return docs


def draw_term(topic):
    prp = (
        """Concatenate the searching keywords (fewer than 5) in the topic with '+'. For example, when given 'experiments about the harmness of using insulin for teenagers to treat diabetes', answer me with 'insulin+harmness+diabetes+teenager'; when given 'The Operational Situation of Bookstores after Digitalization of Reading', answer me with 'bookstore+digitalization'. Topic: """
        + topic
        + ". Give me the answer with no extra words. "
    )
    response = generate_response(prp)
    terms = response.replace("\n", " ").split("+")
    st = ""
    for term in terms[:4]:
        st += term + "+"
    print(st)
    return st[:-1]


def initial(topic, term=None, ret=100):
    if term is None:
        term = draw_term(topic)
    docs = semantic_query(term.replace(" ", "+")[:50], ret)
    sims = dense(topic, docs)
    return docs, sims


def split_question(criteria):
    prp = """Please decompose the complex query into a series of simple questions. You can refer to these examples: [QUERY] Randomized controlled and controlled clinical trials which evaluated individual education for adults with type 2 diabetes. The intervention was individual face-to-face patient education while control individuals received usual care, routine treatment or group education. Only studies that assessed outcome measures at least six months from baseline were included. [QUESTIONS] ###1 Is a randomised controlled trial (RCT) or a controlled clinical trial conducted in the article? ###2 Does the article compare individual face-to-face patient education versus usual care, routine treatment or group education? ###3 Are the patients in the article adults with type 2 diabetes? ###4 Does the article outcome measure at least six months from baseline? [QUERY] Studies were included if they conduct experiments of comparing graph neural networks with other deep learning models. Performance on molecular datasets is reported. [QUESTIONS] ###1 Does the article conduct experiments of graph neural networks? ###2 Does the article compare different deep learning models? ###3 Does the article report performance on molecular datasets?
    Now the QUERY is: 
    """
    prp += criteria
    prp += " What are the decomposed QUESTIONS? "
    response = generate_response(prp)
    orig = response.strip("\n")
    # print('Split Questions: '+orig)
    if orig.find("###") == -1:
        cnt = 1
        while orig.find(str(cnt) + ". ") > -1:
            orig = orig.replace(str(cnt) + ". ", "###" + str(cnt) + " ")
    segs = orig.split("###")
    ques = []
    for seg in segs:
        if len(seg.split(" ")) < 6:
            continue
        if seg[0].isdigit() == False:
            continue
        ques.append(seg[2:])
    return ques


def judge(pred):
    if (
        pred.find("No. ") > -1
        or pred.find("No, ") > -1
        or pred.find("are not") > -1
        or pred.find("is not") > -1
        or pred.find("does not") > -1
        or pred.find("do not") > -1
    ):
        return 0
    else:
        return 1


def check_doc(til, doc, ques):
    string = ""
    for idx, que in enumerate(ques):
        string += str(idx + 1) + ". " + que + " Why? "
    prp = """Read the following article and answer the questions. 
    [Article]

    """
    prp += til + ". " + doc[:10000]
    prp += "[Questions] " + string
    lt = len(prp.split(" "))
    response = generate_response(prp, 4000 - 2 * lt)
    orig = response.strip("\n")
    # print('Title: ', til, ', Check: ', orig.replace('\n', ' '))
    cnt = 2
    preds = []
    orig = " " + orig.replace("\n", " ")
    if orig.find(" 1. ") > -1:
        orig = orig.split(" 1. ")[-1]
        while orig.find(" " + str(cnt) + ". ") > -1:
            pred = orig.split(" " + str(cnt) + ". ")[0]
            preds.append(pred)
            orig = orig.split(" " + str(cnt) + ". ")[-1]
            cnt += 1
        preds.append(orig)
    else:
        preds = [orig]
    sm = 0
    prob = []
    for idx, pred in enumerate(preds):
        if judge(pred):
            sm += 1
        else:
            if idx < len(ques):
                prob.append([ques[idx], pred])
    sm = float(sm) / float(len(preds))
    # print('Score: ', sm)
    return sm, prob


def draw_conclusion(topic, allcon):
    prp = (
        "We want to explore "
        + topic
        + ". Read the analysis table for several articles and summarize your conclusion. # Table: "
        + "\n"
        + allcon
        + "\n # Conclusion: "
    )
    lt = len(prp.split(" "))
    response = generate_response(prp, 4000 - 2 * lt)
    print(allcon)
    print(response)
    return response.replace("\n", " ")


def draw_subtable(pth, topic, docs):
    fw = open(pth, "w")
    cnt = 1
    for ky in docs.keys():
        fw.write("Table #" + str(cnt) + "\n")
        cnt += 1
        prp = (
            "We want to explore "
            + topic
            + ". Read the following article and list the key elements and values of the experiment into a concise markdown table. Article: "
        )
        prp += ky + ". " + docs[ky]
        prp += "Table: "
        lt = len(prp.split(" "))
        response = generate_response(prp, 4000 - 2 * lt)
        fw.write(response.strip("\n"))
        fw.write("\n")
        fw.flush()
    fw.close()


def extra_draw(pth, docs, extra_query):
    fw = open(pth, "w")
    cnt = 1
    for idx, ky in enumerate(docs.keys()):
        fw.write("Table #" + str(cnt) + "\n")
        cnt += 1
        if extra_query[idx] != "":
            prp = extra_query[idx] + ky + ". " + docs[ky]
            prp += "Table: "
            lt = len(prp.split(" "))
            response = generate_response(prp, 4000 - 2 * lt)
            fw.write(response.strip("\n"))
            fw.write("\n")
            fw.flush()
    fw.close()


def deep_reduce(item):
    tag = 0
    item_emb = get_embedding(item.lower())
    sims = []
    for stan in stan_emb:
        sims.append(
            dot(stan, np.array(item_emb))
            / (norm(stan) * norm(np.array(item_emb)) + 1e-5)
        )
    sims = np.array(sims)
    sr = np.argmax(sims)
    if sims[sr] > 0.85:
        item = standard[sr]
        tag = 1
    return item, tag


def convert(table0):
    dic = {}
    table = []
    for tab in table0:
        if tab.find("|") == -1:
            if len(tab.split(":")) == 2:
                tab = "| " + tab.split(":")[0] + " | " + tab.split(":")[1] + " |"
            else:
                continue
        if tab.strip(" ")[0] != "|":
            tab = "|" + tab
        if tab.strip(" ")[-1] != "|":
            tab = tab + "|"
        table.append(tab)
    if len(table) < 2:
        return dic
    segs = table[0].strip("\n").strip(" ")[1:-1].split("|")
    if len(segs) < 2:
        return dic
    tag = 1
    for seg in segs:
        wd, tmptag = deep_reduce(seg)
        if tmptag == 0:
            tag = 0
            break
    if tag and len(table) > 2:
        vals = table[2].strip("\n").strip(" ")[1:-1].split("|")
        if len(vals) != len(segs):
            return {}
        for idx in range(len(segs)):
            wd, tmptag = deep_reduce(segs[idx])
            dic[wd] = vals[idx]
        return dic
    for line in table:
        wds = line.strip("\n").strip(" ")[1:-1].split("|")
        if len(wds) != 2:
            continue
        wd, tmptag = deep_reduce(wds[0])
        if tmptag:
            dic[wd] = wds[1]
    return dic


def combine(tables, tables1=None, crit=None):
    extra_query = []
    if tables1 is not None:
        if len(tables) != len(tables1):
            print("error")
        else:
            for idx in range(len(tables)):
                if tables1[idx] is not None:
                    tables[idx].update(tables1[idx])
    bigtab = {}
    kys = []
    for tab in tables:
        kys += list(tab.keys())
    kys = set(kys)
    if crit is not None:
        for jdx, tab in enumerate(tables):
            prp = ""
            for ky in kys:
                if ky not in tab:
                    prp += ky + ", "
            if prp == "":
                extra_query.append("")
            else:
                ask = (
                    "We want to explore "
                    + crit
                    + ". Read the following article and list the "
                    + prp[:-2]
                    + " of the trial into a concise markdown table. Article: "
                )
                extra_query.append(ask)
    for ky in kys:
        line = " | "
        for tab in tables:
            if ky not in tab:
                line += "N/A"
            else:
                line += tab[ky]
            line += " | "
        bigtab[ky] = line
    return bigtab, extra_query


def get_table(pth, crit):
    lines = open(pth).readlines()[1:]
    tmprec = []
    rec = []
    for line in lines:
        if line[:7] == "Table #":
            rec.append(tmprec)
            tmprec = []
        else:
            tmprec.append(line.strip("\n"))
    rec.append(tmprec)
    dicrec = []
    for tab in rec:
        dicrec.append(convert(tab))
    orig_tab, extra = combine(dicrec, crit=crit)
    return orig_tab, dicrec, extra


def final_table(pth, dicrec):
    lines = open(pth).readlines()[1:]
    tmprec = []
    rec = []
    for line in lines:
        if line[:7] == "Table #":
            rec.append(tmprec)
            tmprec = []
        else:
            tmprec.append(line.strip("\n"))
    rec.append(tmprec)
    dicrec1 = []
    for tab in rec:
        if len(tab) == 0:
            dicrec1.append(None)
        else:
            dicrec1.append(convert(tab))
    final_tab, extra = combine(dicrec, tables1=dicrec1)
    return final_tab


def print_table(tab):
    allcon = ""
    kys = list(tab.keys())
    if len(kys) == 0:
        return allcon
    lt = len(tab[kys[0]].strip(" | ").split("|"))
    header = " | Elements | "
    splitter = " | ----- | "
    for i in range(lt):
        header += str(i + 1) + " | "
        splitter += "---------- | "
    print(header)
    print(splitter)
    allcon += header + "\n"
    allcon += splitter + "\n"
    for ky in kys:
        line = " | " + ky + tab[ky]
        print(line)
        allcon += line + "\n"
    return allcon


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
        "Meta Analysis Plugin",
        description="Analyzing literatures",
        name_for_model="Meta Analysis",
        description_for_model="Plugin for searching and analyzing literatures. All input should be a json like {'input': 'some input'}. Please use the provided questions and search step by step.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )
    QRY = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"

    @tool.get("/search_literature")
    def search_literature(topic: str, maxnum: int, term: str):
        """search for the given topic literatures in the database and return the path of literatures file and the number of literatures. the searching term should be key words in the topic (2-5 words). the number of literatures will be less than maxnum (recommend 30)"""
        topic = topic.replace("AND", "").replace("OR", "")
        if len(topic) < 4:
            term = topic
        else:
            term = term.replace("AND", "").replace("OR", "")
        if len(term.split(" ")) > 4:
            term = None
        retmax = min(maxnum, 50)
        if retmax == 1:
            newdocs, sims = initial(topic, term, 1)
        else:
            docs, sims = initial(topic, term)
            srts = np.argsort(-sims)
            kys = list(docs.keys())
            newdocs = {}
            for sr in srts:
                if sims[sr] < 0.72 and len(newdocs) > 0:
                    break
                if len(newdocs) >= retmax:
                    break
                newdocs[kys[sr]] = docs[kys[sr]]
        pickle.dump(
            newdocs, open("searchdoc_" + topic.replace(" ", "_")[:50] + ".pkl", "wb")
        )
        js = {}
        js["literature_path"] = "searchdoc_" + topic.replace(" ", "_")[:50] + ".pkl"
        js["literature_number"] = len(newdocs)
        return js

    @tool.get("/split_criteria")
    def split_criteria(criteria: str):
        """split the screening requirements in the criteria of the literatures into a series of simple yes/no problems, and return the path of the splitted questions."""
        ques = split_question(criteria)
        np.save("split_ques_" + criteria.replace(" ", "_")[:50] + ".npy", ques)
        js = {
            "question number": len(ques),
            "question_path": "split_ques_" + criteria.replace(" ", "_")[:50] + ".npy",
        }
        return js

    @tool.get("/literature_filter")
    def literature_filter(concat_path: str):
        """Check each literatures saved in the literature path according to the questions saved in the question path, and return the literatures that match the requirements. Concat path is the concatenated string of literature path and question path connected with '&&&'."""
        if len(concat_path.split("&&&")) != 2:
            js = {
                "error": "input path cannot recognize the literature path and the question path. please input 'LITERATURE_PATH&&&QUESTION_PATH'. "
            }
            return js
        liter_pth = concat_path.split("&&&")[0].strip(" ")
        try:
            docs = pickle.load(open(liter_pth, "rb"))
        except Exception as e:
            js = {"error": "cannot open the given literature path!"}
            return js
        ques_pth = concat_path.split("&&&")[1].strip(" ")
        try:
            ques = np.load(ques_pth, allow_pickle=True)
        except Exception as e:
            js = {"error": "cannot open the given question path!"}
            return js
        tmp = []
        for ky in docs:
            sm, prob = check_doc(ky, docs[ky], ques)
            tmp.append(sm)
        tmp = np.array(tmp)
        srts = np.argsort(-tmp)
        finalrecs = []
        kys = list(docs.keys())
        for sr in srts:
            if tmp[sr] < 0.6 and len(finalrecs) > 0:
                break
            if len(finalrecs) > 9:
                break
            finalrecs.append(kys[sr])
        finaldocs = {}
        for ky in finalrecs:
            finaldocs[ky] = docs[ky]
        pickle.dump(finaldocs, open("final_" + liter_pth, "wb"))
        js = {
            "number of matched literatures": len(finaldocs),
            "matched_literature_path": "final_" + liter_pth,
        }
        return js

    @tool.get("/draw_table")
    def draw_table(literature_path_and_topic: str):
        """extract the important elements of the literatures recorded in the literature path and return the path of table records. concatenate the literature path and the analysis topic with '&&&' as the input."""
        if len(literature_path_and_topic.split("&&&")) != 2:
            js = {
                "error": "input path cannot recognize the literature path and the topic. please input 'LITERATURE_PATH&&&TOPIC'. "
            }
            return js
        literature_path, topic = literature_path_and_topic.split("&&&")
        try:
            finaldocs = pickle.load(open(literature_path.strip(" "), "rb"))
        except Exception as e:
            js = {"error": "cannot open the given literature path!"}
            return js
        draw_subtable(
            "subtable_" + literature_path.strip(" ").strip(".pkl") + ".txt",
            topic,
            finaldocs,
        )
        js = {"table_path": "subtable_" + literature_path.strip(".pkl") + ".txt"}
        return js

    @tool.get("/combine_table")
    def combine_table(
        literature_path: str, table_path: str, topic: str
    ):  # ( table_path_and_topic: str ):
        """combine several tables recorded in the table path into one comprehensive record table and return. give the literature path, table path and the exploring topic as the input."""
        # if len(table_path_and_topic.split('&&&'))!=2:
        #    js = {'error': "input path cannot recognize the table path and the topic. please input 'TABLE_PATH&&&TOPIC'. "}
        #    return js
        # table_path, topic = table_path_and_topic.split('&&&')
        try:
            finaldocs = pickle.load(open(literature_path, "rb"))
        except Exception as e:
            js = {"error": "cannot open the given literature path!"}
            return js
        orig_tab, dicrec, extra = get_table(table_path.strip(" "), topic)
        extra_draw("extra_" + table_path.strip(" "), finaldocs, extra)
        final_tab = final_table("extra_" + table_path.strip(" "), dicrec)
        allcon = print_table(final_tab)
        fw = open("big_" + table_path, "w")
        fw.write(allcon)
        fw.close()
        js = {"big table path": "big_" + table_path}
        return js

    @tool.get("/generate_summary")
    def generate_summary(topic: str, table_path: str):
        """given the exploring topic and the record table path of the literatures, this tool generates a paragraph of summary."""
        try:
            table = open(table_path).read()
        except Exception as e:
            js = {"error": "cannot open the table file!"}
            return js
        con = draw_conclusion(topic, table)
        js = {"summary": con}
        return js

    @tool.get("/print_literature")
    def print_literature(literature_path: str, print_num: int):
        """
        given the literature path and number that are required to display, this tool returns the title and abstract of the literature.
        """
        try:
            docs = pickle.load(open(literature_path, "rb"))
        except Exception as e:
            js = {"error": "cannot open the literature file!"}
            return js
        try:
            retmax = max(1, min(5, int(print_num)))
        except Exception as e:
            js = {"error": "illegal number of printing!"}
            return js
        kys = list(docs.keys())[:retmax]
        print_docs = []
        for ky in kys:
            print_docs.append({"Title": ky, "Abstract": docs[ky]})
        js = {"literature_num": len(print_docs), "content": print_docs}
        return js

    @tool.get("/print_tablefile")
    def print_tablefile(table_path: str):
        """
        given the table file path that are required to display, this tool reads the file and returns the string of the table.
        """
        try:
            con = open(table_path).read()
        except Exception as e:
            js = {"error": "cannot open the table file!"}
            return js
        js = {"table string": con}
        return js

    return tool
