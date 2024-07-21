#pragma once
#include "DoubleYearThresholdCPUFunc.cpp"


const int __SAVE_FREQ__ = 300;


class DoubleYearThresholdFilter: public Generator {
public:
    double *d_threshold;
    double *d_result;
    double *h_final;

    chrono::high_resolution_clock::time_point start;

    DoubleYearThresholdFilter(string config_path);
    ~DoubleYearThresholdFilter();

    bool compute_result(bool force_save);
};


DoubleYearThresholdFilter::DoubleYearThresholdFilter(string config_path)
: Generator(config_path) {
    int num_threshold = __NUM_THRESHOLD_PER_CYCLE__*(index_length - 2);
    d_threshold = new double[(config.storage_size+cols)*num_threshold];
    d_result = new double[2*(config.storage_size+cols)*num_threshold*config.num_cycle];
    h_final = new double[4*(config.storage_size+cols)*config.num_cycle];
    cuda_set_array_value(
        d_result, 2*(config.storage_size+cols)*num_threshold*config.num_cycle, 0
    );
    h_final = new double[4*(config.storage_size+cols)*config.num_cycle];

    start = chrono::high_resolution_clock::now();
}


DoubleYearThresholdFilter::~DoubleYearThresholdFilter(){
    delete[] d_threshold;
    delete[] d_result;
    delete[] h_final;
}


bool DoubleYearThresholdFilter::compute_result(bool force_save){
    fill_thresholds(
        temp_weight_storage, d_threshold, INDEX, index_length, count_temp_storage, rows
    );

    int num_threshold = __NUM_THRESHOLD_PER_CYCLE__*(index_length - 2);
    double_year_threshold_investing(
        temp_weight_storage, d_threshold, d_result, count_temp_storage, num_threshold,
        rows, config.num_cycle, config.interest, INDEX, PROFIT, SYMBOL, BOOL_ARG, index_length
    );

    find_best_results(
        d_result, d_threshold, h_final, count_temp_storage, num_threshold, config.num_cycle
    );

    //
    int i, j;
    int64_t k;
    int num_field = config.filter_field.size();
    for (i=0; i<count_temp_storage; i++){
        for (j=0; j<config.num_cycle; j++){
            if (h_final[i*config.num_cycle*num_field + j*num_field + config.eval_index]
                >= config.eval_threshold){
                k = current[2][0]+i;
                update_insert_query(queries[j],
                                    k,
                                    fml_shape/2,
                                    temp_formula_storage[i],
                                    cols,
                                    num_field,
                                    h_final + i*config.num_cycle*num_field + j*num_field);
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

    if (save){
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
