import pandas as pd
import numpy as np
import os
import json
from colorama import Fore, Style
import time
import multiprocessing
from PyScripts import suppFunc


def run_worker(lib_abs_path, generate_method, filter_name, worker_type, config_path, wait_before_run):
    time.sleep(wait_before_run)

    command = f"{lib_abs_path}ExeFile/"

    command += suppFunc.generate_method_name[generate_method]["command"]

    command += suppFunc.filter_fields[filter_name]["command"]

    if worker_type == "CPU":
        command += "CPU.exe"
    elif worker_type == "GPU":
        command += "CUDA.exe"
    else: raise

    command += f" {config_path}"

    print(Fore.LIGHTCYAN_EX + "Starting worker with input:",
          Fore.LIGHTMAGENTA_EX + "; " + lib_abs_path + "; " + generate_method
          + "; " +filter_name + "; " +worker_type + "; " +config_path, Style.RESET_ALL)
    print(command)
    print()
    os.system(f"start /wait cmd /c {command}")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    num_worker = config[0]["num_worker"]
    worker_type = config[0]["worker_type"]
    timeout_per_task = config[0]["timeout_per_task"]
    warehouse_path = config[0]["warehouse_path"]

    assert num_worker in [1, 2, 3]
    assert worker_type in ["GPU", "CPU", "Hybrid"]
    assert timeout_per_task >= 1
    assert not (num_worker == 1 and worker_type == "Hybrid")
    assert not (num_worker == 3 and worker_type == "GPU")
    assert not (num_worker != 1 and worker_type == "CPU")

    lib_abs_path = __file__.replace("main.py", "")
    list_config_path = []
    list_generate_method = []
    list_filter_name = []
    for i in range(1, len(config)):
        task_config = []

        # prepare data and data_path
        data_path = config[i]["data_path"]
        assert "\\" not in data_path
        data = pd.read_excel(data_path)
        data_name = data_path.split("/")[-1].replace(".xlsx", "")
        if data_name not in os.listdir(warehouse_path):
            os.makedirs(warehouse_path + f"/{data_name}")

        try:
            data_full = pd.read_excel(warehouse_path + f"/{data_name}" + f"/data_full.xlsx")
            print(Fore.GREEN + "Read data_full:", warehouse_path + f"/{data_name}" + f"/data_full.xlsx", Style.RESET_ALL)
        except:
            data.to_excel(warehouse_path + f"/{data_name}" + f"/data_full.xlsx", index=False)
            data_full = pd.read_excel(warehouse_path + f"/{data_name}" + f"/data_full.xlsx")
            print(Fore.GREEN + "Created data_full:", warehouse_path + f"/{data_name}" + f"/data_full.xlsx", Style.RESET_ALL)

        suppFunc.compare_dfs(data, data_full)

        try:
            data_train = pd.read_excel(warehouse_path + f"/{data_name}" + f"/data_train.xlsx")
            print(Fore.GREEN + "Read data_train:", warehouse_path + f"/{data_name}" + f"/data_train.xlsx", Style.RESET_ALL)
        except:
            max_cycle = max(data_full["TIME"])
            data_train = data_full[data_full["TIME"] < max_cycle].reset_index(drop=True)
            data_train.index += 1
            for col in data_train.columns:
                if data_train[col].dtype == "object":
                    data_train.loc[0, col] = "_NULL_"
                else:
                    data_train.loc[0, col] = 0

            data_train.loc[0, "TIME"] = max_cycle
            data_train.sort_index(inplace=True)
            data_train.to_excel(warehouse_path + f"/{data_name}" + f"/data_train.xlsx", index=False)
            print(Fore.GREEN + "Created data_train:", warehouse_path + f"/{data_name}" + f"/data_train.xlsx", Style.RESET_ALL)
            data_train = pd.read_excel(warehouse_path + f"/{data_name}" + f"/data_train.xlsx")

        print()
        task_config.append(f"data_path = {warehouse_path}/{data_name}/data_train.xlsx")

        # filter_field
        filter_name = config[i]["filter"]
        list_filter_name.append(filter_name)
        filter_fields = suppFunc.filter_fields[filter_name]["fields"]
        task_config.append(f"filter_field = {filter_fields}")

        # interest, valuearg_threshold
        interest = config[i]["interest"]
        task_config.append(f"interest = {interest}")
        valuearg_threshold = config[i]["valuearg_threshold"]
        task_config.append(f"valuearg_threshold = {valuearg_threshold}")

        # folder_save
        generate_method = config[i]["generate_method"]
        assert generate_method in list(suppFunc.generate_method_name.keys())
        list_generate_method.append(generate_method)

        folder_save = f"{warehouse_path}/{data_name}/{generate_method}/{filter_name}"
        os.makedirs(folder_save, exist_ok=True)
        task_config.append(f"folder_save = {folder_save}")

        # eval_index, eval_threshold
        eval_index = config[i]["eval_index"]
        eval_threshold = config[i]["eval_threshold"]
        task_config.append(f"eval_index = {eval_index}")
        task_config.append(f"eval_threshold = {eval_threshold}")

        # storage_size
        try:
            storage_size = config[i]["temp_storage_size"]
        except:
            storage_size = 1000
            print(Fore.YELLOW + "If storage_size is not specified, the default is 1000", Style.RESET_ALL)

        task_config.append(f"storage_size = {storage_size}")

        # num_cycle
        num_cycle = config[i]["num_cycle"]
        task_config.append(f"num_cycle = {num_cycle}")

        # lib_abs_path
        task_config.append(f"lib_abs_path = {lib_abs_path}")

        # timeout_in_minutes
        task_config.append(f"timeout_in_minutes = {timeout_per_task}")

        # Save task_config
        config_path = f"{folder_save}/config.txt"
        list_config_path.append(config_path)
        with open(config_path, "w") as f:
            f.write("\n".join(task_config))
            print(Fore.GREEN + "Created task_config:", config_path, Style.RESET_ALL)
            print()
            print("\n".join(task_config))
            print()

    if num_worker == 1:
        list_worker_type = [worker_type]
    elif num_worker == 2:
        if worker_type == "Hybrid":
            list_worker_type = ["GPU", "CPU"]
        else:
            list_worker_type = ["GPU", "GPU"]
    else:
        list_worker_type = ["GPU", "GPU", "CPU"]

    n = len(list_config_path)
    while True:
        for i in range(n):
            list_worker_input = []
            for j in range(num_worker):
                generate_method = list_generate_method[(i+j)%n]
                filter_name = list_filter_name[(i+j)%n]
                worker_type = list_worker_type[j]
                config_path = list_config_path[(i+j)%n]

                list_worker_input.append((
                    lib_abs_path,
                    generate_method,
                    filter_name,
                    worker_type,
                    config_path,
                    j
                ))

            pool = multiprocessing.Pool(processes=num_worker)
            pool.starmap(run_worker, list_worker_input)
            pool.close()
            pool.join()
            time.sleep(5)

        # Tong ket cong thuc o day
        command = "python " + lib_abs_path + "PyScripts/query_data_formula.py "\
            + config[0]["folder_formula"] + " "\
            + config[0]["warehouse_path"] + " "

        for k in range(n):
            command += list_config_path[k] + " "

        os.system(command)

    #_________________________END_________________________#