import psycopg2
import pymysql
import json
import logging
import os
from enum import IntEnum


class DataType(IntEnum):
    VALUE = 0
    TIME = 1
    CHAR = 2


AGGREGATE_CONSTRAINTS = {
    DataType.VALUE.value: ["count", "max", "min", "avg", "sum"],
    DataType.VALUE.CHAR: ["count", "max", "min"],
    DataType.VALUE.TIME: ["count", "max", "min"],
}


def transfer_field_type(database_type, server):
    data_type = list()
    if server == "mysql":
        data_type = [
            [
                "int",
                "tinyint",
                "smallint",
                "mediumint",
                "bigint",
                "float",
                "double",
                "decimal",
            ],
            ["date", "time", "year", "datetime", "timestamp"],
        ]
        database_type = database_type.lower().split("(")[0]
    elif server == "postgresql":
        data_type = [["integer", "numeric"], ["date"]]
    if database_type in data_type[0]:
        return DataType.VALUE.value
    elif database_type in data_type[1]:
        return DataType.TIME.value
    else:
        return DataType.CHAR.value


class DBArgs(object):
    def __init__(self, dbtype, config, dbname=None):
        self.dbtype = dbtype
        if self.dbtype == "mysql":
            self.host = config["host"]
            self.port = config["port"]
            self.user = config["user"]
            self.password = config["password"]
            self.dbname = dbname if dbname else config["dbname"]
            self.driver = "com.mysql.jdbc.Driver"
            self.jdbc = "jdbc:mysql://"
        else:
            self.host = config["host"]
            self.port = config["port"]
            self.user = config["user"]
            self.password = config["password"]
            self.dbname = dbname if dbname else config["dbname"]
            self.driver = "org.postgresql.Driver"
            self.jdbc = "jdbc:postgresql://"


class Database:
    def __init__(self, args, timeout=-1):
        self.args = args
        self.conn = self.resetConn(timeout)

        # self.schema = self.compute_table_schema()

    def resetConn(self, timeout=-1):
        if self.args.dbtype == "mysql":
            conn = pymysql.connect(
                host=self.args.host,
                user=self.args.user,
                passwd=self.args.password,
                database=self.args.dbname,
                port=int(self.args.port),
                charset="utf8",
                connect_timeout=timeout,
                read_timeout=timeout,
                write_timeout=timeout,
            )
        else:
            if timeout > 0:
                conn = psycopg2.connect(
                    database=self.args.dbname,
                    user=self.args.user,
                    password=self.args.password,
                    host=self.args.host,
                    port=self.args.port,
                    options="-c statement_timeout={}s".format(timeout),
                )
            else:
                conn = psycopg2.connect(
                    database=self.args.dbname,
                    user=self.args.user,
                    password=self.args.password,
                    host=self.args.host,
                    port=self.args.port,
                )

        return conn

    """
    def exec_fetch(self, statement, one=True):
        cur = self.conn.cursor()
        cur.execute(statement)
        if one:
            return cur.fetchone()
        return cur.fetchall()    
    """

    def execute_sql(self, sql):
        fail = 1
        self.conn = self.resetConn()
        cur = self.conn.cursor()
        i = 0
        cnt = 3  # retry times
        while fail == 1 and i < cnt:
            try:
                fail = 0
                cur.execute(sql)
            except BaseException:
                fail = 1
            res = []
            if fail == 0:
                res = cur.fetchall()
            i = i + 1
        logging.debug(
            "database {}, return flag {}, execute sql {}\n".format(
                self.args.dbname, 1 - fail, sql
            )
        )
        if fail == 1:
            # raise RuntimeError("Database query failed")
            print("SQL Execution Fatal!!")
            return 0, ""
        elif fail == 0:
            # print("SQL Execution Succeed!!")
            return 1, res

    def pgsql_results(self, sql):
        try:
            # success, res = self.execute_sql('explain (FORMAT JSON, analyze) ' + sql)
            success, res = self.execute_sql(sql)
            # print("pgsql_results", success, res)
            if success == 1:
                return res
            else:
                return "<fail>"
        except Exception as error:
            logging.error("pgsql_results Exception", error)
            return "<fail>"

    def pgsql_cost_estimation(self, sql):
        try:
            # success, res = self.execute_sql('explain (FORMAT JSON, analyze) ' + sql)
            success, res = self.execute_sql("explain (FORMAT JSON) " + sql)
            if success == 1:
                cost = res[0][0][0]["Plan"]["Total Cost"]
                return cost
            else:
                logging.error("pgsql_cost_estimation Fails!")
                return 0
        except Exception as error:
            logging.error("pgsql_cost_estimation Exception", error)
            return 0

    def pgsql_actual_time(self, sql):
        try:
            # success, res = self.execute_sql('explain (FORMAT JSON, analyze) ' + sql)
            success, res = self.execute_sql("explain (FORMAT JSON, analyze) " + sql)
            if success == 1:
                cost = res[0][0][0]["Plan"]["Actual Total Time"]
                return cost
            else:
                return -1
        except Exception as error:
            logging.error("pgsql_actual_time Exception", error)
            return -1

    def mysql_cost_estimation(self, sql):
        try:
            success, res = self.execute_sql("explain format=json " + sql)
            if success == 1:
                total_cost = self.get_mysql_total_cost(0, json.loads(res[0][0]))
                return float(total_cost)
            else:
                return -1
        except Exception as error:
            logging.error("mysql_cost_estimation Exception", error)
            return -1

    def get_mysql_total_cost(self, total_cost, res):
        if isinstance(res, dict):
            if "query_cost" in res.keys():
                total_cost += float(res["query_cost"])
            else:
                for key in res:
                    total_cost += self.get_mysql_total_cost(0, res[key])
        elif isinstance(res, list):
            for i in res:
                total_cost += self.get_mysql_total_cost(0, i)

        return total_cost

    def get_tables(self):
        if self.args.dbtype == "mysql":
            return self.mysql_get_tables()
        else:
            return self.pgsql_get_tables()

    # query cost estimated by the optimizer
    def cost_estimation(self, sql):
        if self.args.dbtype == "mysql":
            return self.mysql_cost_estimation(sql)
        else:
            return self.pgsql_cost_estimation(sql)

    def compute_table_schema(self):
        """
        schema: {table_name: [field_name]}
        :param cursor:
        :return:
        """

        if self.args.dbtype == "postgresql":
            # cur_path = os.path.abspath('.')
            # tpath = cur_path + '/sampled_data/'+dbname+'/schema'
            sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            success, res = self.execute_sql(sql)
            # print("======== tables", res)
            if success == 1:
                tables = res
                schema = {}
                for table_info in tables:
                    table_name = table_info[0]
                    sql = (
                        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '"
                        + table_name
                        + "';"
                    )
                    success, res = self.execute_sql(sql)
                    # print("======== table columns", res)
                    columns = res
                    schema[table_name] = []
                    for col in columns:
                        """compute the distinct value ratio of the column

                        if transfer_field_type(col[1], self.args.dbtype) == DataType.VALUE.value:
                            sql = 'SELECT count({}) FROM {};'.format(col[0], table_name)
                            success, res = self.execute_sql(sql)
                            print("======== column rows", res)
                            num = res
                            if num[0][0] != 0:
                                schema[table_name].append(col[0])
                        """

                        # schema[table_name].append("column {} is of {} type".format(col[0], col[1]))
                        schema[table_name].append("{}".format(col[0]))
                """
                with open(tpath, 'w') as f:
                    f.write(str(schema))
                """
                # print(schema)
                return schema

            else:
                logging.error("pgsql_cost_estimation Fails!")
                return 0

    def simulate_index(self, index):
        # table_name = index.table()
        statement = "SELECT * FROM hypopg_create_index(E'{}');".format(index)
        result = self.execute_sql(statement)

        return result

    def drop_simulated_index(self, oid):
        statement = f"select * from hypopg_drop_index({oid})"
        result = self.execute_sql(statement)

        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."
