import re
import sqlparse


def remove_create_table(sql):
    return re.sub(
        r"(create|CREATE)\s+(table|TABLE).+?\(.+?\)\s*;", "", sql, flags=re.DOTALL
    )


def remove_create_index(sql):
    return re.sub(
        r"(create|CREATE)\s+(index|INDEX).+?\(.+?\)\s*;", "", sql, flags=re.DOTALL
    )


def remove_table(sql):
    return re.sub(r"(table|TABLE).+?\(.+?\)\s*;", "", sql, flags=re.DOTALL)


def clean_sql(sql):
    tmp = []
    for token in sql.flatten():
        if (
            not token.is_whitespace
            and not token.ttype is sqlparse.tokens.Comment.Single
        ):
            tmp.append(token)
    return strip_par(" ".join(str(t) for t in tmp))


def strip_par(s):
    for op in ["(", ")", ",", ">", "=", "<", ">=", "<=", "!=", "<>", ".", ";"]:
        s = s.replace(" {}".format(op), op).replace("{} ".format(op), op)
    return s


def preprocess_execute_sql(sql):
    sql = remove_create_table(sql)
    sql = remove_create_index(sql)
    parsed = sqlparse.parse(sql)
    if len(parsed) == 0:
        return [0, ""]
    sql = clean_sql(parsed[0])
    if not sql:
        return [0, ""]
    if sql[-1] != ";":
        sql += ";"
    return [1, sql]
