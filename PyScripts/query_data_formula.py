from sqlite3 import Connection
from numpy import array, unique
from pandas import read_excel
from pandas import DataFrame
from json import load
from sys import argv
from os import makedirs
from multiprocessing import Pool
from base import Base, decode_formula, convert_arrF_to_strF, convert_strF_to_arrF, similarity_filter
import suppFunc
from colorama import Fore, Style
from detail_formula import process


def get_list_table():
    return '''SELECT name FROM sqlite_master WHERE type = "table";'''


def _top_n_by_column(table_name, column, n_row, select_all):
    num_operand = int(table_name.split("_")[1])
    text_1 = '"id"'
    for i in range(num_operand):
        text_1 += f', "E{i}"'

    if select_all:
        text_1 = "*"
    else:
        text_1 += f', "{column}"'

    return f'SELECT {text_1} FROM "{table_name}" ORDER BY "{column}" DESC LIMIT {n_row};'


def top_n_by_column(time, column, n_row, db_file_path, num_data_operand, select_all=False):
    connection = Connection(db_file_path)
    cursor = connection.cursor()
    cursor.execute(get_list_table())
    list_table = [t_[0] for t_ in cursor.fetchall() if t_[0].startswith("T"+str(time))]
    list_of_list_value = []

    for table in list_table:
        query = _top_n_by_column(table, column, n_row, select_all)
        cursor.execute(query)
        list_value = cursor.fetchall()
        n_op = int(table.split("_")[1])
        for i in range(len(list_value)):
            temp = array(list(list_value[i][1:n_op+1]))
            ct = decode_formula(temp, num_data_operand).astype(int)
            list_value[i] = [list_value[i][0]] + [convert_arrF_to_strF(ct)] + list(list_value[i][n_op+1:])
        list_of_list_value += list_value

    if select_all:
        list_col = ["id", "CT"]
        cursor.execute(f"PRAGMA table_info('{table}');")
        for r_ in cursor.fetchall()[n_op+1:]:
            list_col.append(r_[1])
    else:
        list_col = ["id", "CT", column]

    data = DataFrame(list_of_list_value, columns=list_col)
    data.sort_values(column, inplace=True, ignore_index=True, ascending=False)
    connection.close()
    return data.loc[:n_row-1]


if __name__ == "__main__":
    folder_path = argv[1]
    warehouse_p = argv[2]
    list_config_path = argv[3:]
    for i in range(len(list_config_path)):
        # Dest folder
        config_path = list_config_path[i]
        dest_folder = config_path.replace(warehouse_p, folder_path).replace("config.txt", "")
        print(dest_folder)
        makedirs(dest_folder, exist_ok=True)

        # Read config
        with open(config_path, "r") as f:
            text_config = f.read()

        config = {}
        for line in text_config.split("\n"):
            a, b = line.split("=")
            config[a.strip()] = b.strip()

        # Read data_full to find num_data_operand, list_time
        data_full = read_excel(config["data_path"])
        vis = Base(data_full, float(config["interest"]), float(config["valuearg_threshold"]))
        num_data_operand = vis.OPERAND.shape[0]

        db_path = config_path.replace("config.txt", "f.db")
        connection = Connection(db_path)
        cursor = connection.cursor()
        cursor.execute(get_list_table())
        list_table = [t_[0] for t_ in cursor.fetchall() if t_[0].startswith("T")]
        list_time = [int(t.replace("T", "").split("_")[0]) for t in list_table]
        list_time = unique(list_time)
        print(list_time)
        connection.close()

        # Tim cot de sap xep
        diff_time = data_full["TIME"].max() - max(list_time)
        column = suppFunc.filter_fields[config["folder_save"].split("/")[-1]]["fields"].split(";")[int(config["eval_index"])]
        print(column)

        # Query 10x needed formulas
        list_args = []
        for t_ in list_time:
            list_args.append((
                t_,
                column,
                1000000,
                db_path,
                num_data_operand,
                False
            ))

        pool = Pool(processes=len(list_args))
        results = pool.starmap(top_n_by_column, list_args)
        pool.close()
        pool.join()

        # similarity filter
        list_args = []
        for i in range(len(results)):
            list_args.append((
                results[i],
                "CT",
                10000,
                2
            ))

        pool = Pool(processes=len(list_args))
        results = pool.starmap(similarity_filter, list_args)
        pool.close()
        pool.join()

        # Detail formula
        list_args = []
        for i in range(len(results)):
            list_args.append((
                data_full.copy(),
                results[i],
                float(config["interest"]),
                float(config["valuearg_threshold"]),
                i + diff_time
            ))

        pool = Pool(processes=len(list_args))
        results = pool.starmap(process, list_args)
        pool.close()
        pool.join()

        # Save df
        for i in range(len(results)):
            results[i].to_csv(dest_folder + f"{i+diff_time}.csv", index=False)
            print(Fore.LIGHTGREEN_EX + "Saved:", dest_folder + f"{i+diff_time}.csv", Style.RESET_ALL)
