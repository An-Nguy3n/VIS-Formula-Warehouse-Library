#pragma once
#include "NoBalanceValueCPUFunc.cpp"
#include "../../Generator/HomoPoly/CPU/HomoPolyMethod.cpp"


const int __SAVE_FREQ__ = 300;


class NoBalanceValueFilter: public Generator {
public:
    double *d_profits;
    double *d_profits_origin;
    double *h_results;

    double *d_temp_wgt;
    double *d_temp_prf;

    chrono::high_resolution_clock::time_point start;
    NoBalanceValueFilter(string config_path);
    ~NoBalanceValueFilter();

    bool compute_result(bool force_save);
};


NoBalanceValueFilter::NoBalanceValueFilter(string config_path)
: Generator(config_path) {
    d_profits = new double[rows*(config.storage_size+cols)];
    d_profits_origin = new double[rows*(config.storage_size+cols)];
    h_results = new double[config.num_cycle*(config.storage_size+cols)];

    d_temp_wgt = new double[rows*(config.storage_size+cols)];
    d_temp_prf = new double[rows*(config.storage_size+cols)];

    start = chrono::high_resolution_clock::now();

    for (int i=0; i<config.storage_size+cols; i++)
        memcpy(d_profits_origin+i*rows, PROFIT, 8*rows);
}


NoBalanceValueFilter::~NoBalanceValueFilter(){
    delete[] d_profits;
    delete[] d_profits_origin;
    delete[] h_results;
    delete[] d_temp_wgt;
    delete[] d_temp_prf;
}


bool NoBalanceValueFilter::compute_result(bool force_save){
    memcpy(d_profits, d_profits_origin, 8*rows*count_temp_storage);

    get_noBalance_value(
        temp_weight_storage,
        d_profits,
        d_temp_wgt,
        d_temp_prf,
        rows,
        INDEX,
        index_length,
        config.num_cycle,
        h_results,
        count_temp_storage
    );

    //
    int i, j;
    int64_t k;
    for (i=0; i<count_temp_storage; i++){
        for (j=0; j<config.num_cycle; j++){
            if (h_results[i*config.num_cycle + j] >= config.eval_threshold){
                k = current[2][0] + i;
                update_insert_query(queries[j],
                                    k,
                                    fml_shape/2,
                                    temp_formula_storage[i],
                                    cols,
                                    1,
                                    h_results + i*config.num_cycle + j);
            }
        }
    }

    current[2][0] += count_temp_storage;
    count_temp_storage = 0;

    bool save = false;
    if (force_save) save = true;
    if (!save){
        for (i=0; i<config.num_cycle; i++){
            if (queries[i].size() >= 1000000){
                save = true; break;
            }
        }
    }
    if (!save){
        chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();
        chrono::seconds duration = chrono::duration_cast<chrono::seconds>(now - start);
        if (duration.count() >= __SAVE_FREQ__) save = true;
    }

    if (save) {
        string all_query;
        all_query.reserve(config.num_cycle*1000000);
        for (i=0; i<config.num_cycle; i++){
            complete_insert_query(queries[i]);
            if (queries[i].substr(queries[i].size()-7, 6) != "values")
                all_query += queries[i];
        }
        all_query += "delete from checkpoint_" + to_string(fml_shape/2) + ";";
        all_query += "insert into checkpoint_" + to_string(fml_shape/2) + " values (";
        all_query += to_string(current[2][0]) + "," + to_string(current[1][0]) + ",";
        for (i=0; i<fml_shape; i++){
            all_query += to_string(current[0][i]) + ",";
        }
        all_query.pop_back();
        all_query += ");";

        string filename = config.folder_save + "/queries.bin";
        ofstream outFile(filename.c_str(), ios::binary);
        if (outFile.is_open()){
            outFile.write(all_query.c_str(), all_query.size());
            outFile.close();
        }
        else raise_error("Khong mo duoc file", filename);

        string command = "python " + config.lib_abs_path + "/PyScripts/run_query.py "
            + config.folder_save + "/f.db";
        system(command.c_str());

        start = chrono::high_resolution_clock::now();
        for (i=0; i<config.num_cycle; i++){
            queries[i] = init_insert_query(fml_shape/2, i);
            queries[i].reserve(1000000);
        }

        chrono::high_resolution_clock::time_point check_timeout = chrono::high_resolution_clock::now();
        chrono::seconds total_runtime = chrono::duration_cast<chrono::seconds>(check_timeout - time_start);
        long long time_range = total_runtime.count();

        cout << FG_CYAN << "Running: " << time_range << "/" << config.timeout_in_minutes*60 << " secs!\n" << RESET_COLOR;
        cout << FG_BRIGHT_CYAN << current[2][0] << endl;
        cout << current[1][0] << endl;
        for (i=0; i<2*num_opr_per_fml; i++) cout << current[0][i] << " ";
        cout << endl << RESET_COLOR;

        if (time_range >= config.timeout_in_minutes*60) return true;
    }
    return false;
}
