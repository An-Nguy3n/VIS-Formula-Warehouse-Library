#pragma once
#include "HomoPolyKernel.cu"
#include "../HomoPolyConfig.cpp"
#include "../../../Utils/WorkWithFile.cpp"
#include "../../../Utils/SuppFunc.cpp"
#include <chrono>


class Generator {
public:
    HomoPolyConfig config;
    int64_t **current, num_opr_per_fml;

    int *INDEX, *SYMBOL, *BOOL_ARG, index_length, rows, cols;
    double *PROFIT, *OPERAND;

    double *temp_weight_storage;
    int count_temp_storage;
    uint8_t **temp_formula_storage;

    chrono::high_resolution_clock::time_point time_start;

    //
    int fml_shape, groups, num_per_grp;

    //
    string *queries;

    //
    Generator(string config_path);
    ~Generator();

    bool fill_formula(uint8_t *formula, int **f_struct, int idx,
        double *temp_0, int temp_op, double *temp_1,
        int mode, bool add_sub, bool mul_div
    );

    virtual bool compute_result(bool force_save);
    void run();
};


Generator::Generator(string config_path){
    config = HomoPolyConfig(config_path);

    // Extract data
    int *_INDEX, *_SYMBOL, *_BOOL_ARG;
    double *_PROFIT, *_OPERAND;

    string command = "python " + config.lib_abs_path + "/PyScripts/extract_data.py "
        + config.data_path + " " + to_string(config.interest) + " "
        + to_string(config.valuearg_threshold) + " " + config.folder_save;
    system(command.c_str());

    // Load data
    read_binary_file_1d(_INDEX, index_length, config.folder_save + "/InputData/INDEX.bin");
    read_binary_file_1d(_SYMBOL, rows, config.folder_save + "/InputData/SYMBOL.bin");
    read_binary_file_1d(_BOOL_ARG, rows, config.folder_save + "/InputData/BOOL_ARG.bin");
    read_binary_file_1d(_PROFIT, rows, config.folder_save + "/InputData/PROFIT.bin");
    read_binary_file_2d(_OPERAND, cols, rows, config.folder_save + "/InputData/OPERAND.bin");

    cudaMalloc((void**)&INDEX, 4*index_length);
    cudaMalloc((void**)&SYMBOL, 4*rows);
    cudaMalloc((void**)&BOOL_ARG, 4*rows);
    cudaMalloc((void**)&PROFIT, 8*rows);
    cudaMalloc((void**)&OPERAND, 8*rows*cols);
    cudaMemcpy(INDEX, _INDEX, 4*index_length, cudaMemcpyHostToDevice);
    cudaMemcpy(SYMBOL, _SYMBOL, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(BOOL_ARG, _BOOL_ARG, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(PROFIT, _PROFIT, 8*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(OPERAND, _OPERAND, 8*rows*cols, cudaMemcpyHostToDevice);
    delete[] _INDEX;
    delete[] _SYMBOL;
    delete[] _BOOL_ARG;
    delete[] _PROFIT;
    delete[] _OPERAND;

    // Load checkpoint
    load_checkpoint_PolyMethod(current, config.folder_save, num_opr_per_fml, config.lib_abs_path);

    // Init storage
    cudaMalloc((void**)&temp_weight_storage, 8*(config.storage_size+cols)*rows);
    temp_formula_storage = new uint8_t*[config.storage_size+cols];
    count_temp_storage = 0;

    //
    time_start = chrono::high_resolution_clock::now();
    queries = new string[config.num_cycle];
}


Generator::~Generator(){
    cudaFree(INDEX);
    cudaFree(SYMBOL);
    cudaFree(BOOL_ARG);
    cudaFree(PROFIT);
    cudaFree(OPERAND);

    for (int i=0; i<3; i++) delete[] current[i];
    delete[] current;

    cudaFree(temp_weight_storage);
    delete[] temp_formula_storage;

    delete[] queries;

    chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();
    chrono::duration<long long> duration = chrono::duration_cast<chrono::seconds>(now - time_start);
    cout << FG_GREEN << "Thoi gian Worker chay: " << duration.count() << " seconds.\n" << RESET_COLOR;
}


bool Generator::fill_formula(
    uint8_t *formula,
    int **f_struct,
    int idx,
    double *temp_0,
    int temp_op,
    double *temp_1,
    int mode,
    bool add_sub,
    bool mul_div
) {
    if (!mode) /*Sinh dau cong tru*/ {
        int gr_idx = 2147483647, start = 0, i, k;
        bool new_add_sub;
        uint8_t *new_formula = new uint8_t[fml_shape];
        int **new_f_struct = new int*[groups];
        for (i=0; i<groups; i++) new_f_struct[i] = new int[4];

        // Xac dinh nhom
        for (i=0; i<groups; i++){
            if (f_struct[i][2]-1 == idx){
                gr_idx = i;
                break;
            }
        }

        // Xac dinh chi so bat dau
        if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];

        // Loop
        for (k=start; k<2; k++){
            memcpy(new_formula, formula, fml_shape);
            for (i=0; i<groups; i++) memcpy(new_f_struct[i], f_struct[i], 16);
            new_formula[idx] = k;
            new_f_struct[gr_idx][0] = k;
            if (k == 1){
                new_add_sub = true;
                for (i=gr_idx+1; i<groups; i++){
                    new_formula[new_f_struct[i][2]-1] = 1;
                    new_f_struct[i][0] = 1;
                }
            } else new_add_sub = false;

            if (fill_formula(new_formula, new_f_struct, idx+1,
                            temp_0, temp_op, temp_1, 1, new_add_sub, mul_div)) return true;
        }

        // Giai phong bo nho
        delete[] new_formula;
        for (i=0; i<groups; i++) delete[] new_f_struct[i];
        delete[] new_f_struct;
    }
    else if (mode == 2) /*Sinh dau nhan chia*/ {
        int start = 2, i, j, k;
        bool new_mul_div;
        uint8_t *new_formula = new uint8_t[fml_shape];
        int **new_f_struct = new int*[groups];
        for (i=0; i<groups; i++) new_f_struct[i] = new int[4];

        // Xac dinh chi so bat dau
        if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];
        if (!start) start = 2;

        // Loop
        bool *valid_operator = get_valid_operator(f_struct, idx, start);
        for (k=0; k<2; k++){
            if (!valid_operator[k]) continue;
            memcpy(new_formula, formula, fml_shape);
            for (i=0; i<groups; i++) memcpy(new_f_struct[i], f_struct[i], 16);
            new_formula[idx] = k + 2;
            if (k == 1){
                new_mul_div = true;
                for (i=idx+2; i<2*new_f_struct[0][1]-1; i+=2) new_formula[i] = 3;
                for (i=1; i<groups; i++){
                    for (j=0; j<new_f_struct[0][1]-1; j++)
                        new_formula[new_f_struct[i][2] + 2*j + 1] = new_formula[2 + 2*j];
                }
            } else {
                new_mul_div = false;
                for (i=0; i<groups; i++) new_f_struct[i][3] += 1;
                if (idx == 2*new_f_struct[0][1]-2){
                    new_mul_div = true;
                    for (i=1; i<groups; i++){
                        for (j=0; j<new_f_struct[0][1]-1; j++)
                            new_formula[new_f_struct[i][2] + 2*j + 1] = new_formula[2 + 2*j];
                    }
                }
            }

            if (fill_formula(new_formula, new_f_struct, idx+1,
                            temp_0, temp_op, temp_1, 1, add_sub, new_mul_div)) return true;
        }

        // Giai phong bo nho
        delete[] valid_operator;
        delete[] new_formula;
        for (i=0; i<groups; i++) delete[] new_f_struct[i];
        delete[] new_f_struct;
    }
    else if (mode == 1){
        int start = 0, i, count = 0;

        // Xac dinh chi so bat dau
        if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];

        // Xac dinh cac toan hang hop le va dem
        bool *valid_operand = get_valid_operand(formula, f_struct, idx, start, cols, groups);
        for (i=0; i<cols; i++){
            if (valid_operand[i]) count++;
        }

        if (count){
            int temp_op_new, new_idx, new_mode, k = 0;
            bool chk = false, temp_0_change;
            double *new_temp_0 = nullptr, *new_temp_1 = nullptr;
            uint8_t **new_formula = new uint8_t*[count];
            int *valid = new int[count];
            int *d_valid;
            int num_block = count*rows/256 + 1;
            cudaMalloc((void**)&d_valid, 4*count);

            for (i=0; i<cols; i++){
                if (!valid_operand[i]) continue;
                new_formula[k] = new uint8_t[fml_shape];
                memcpy(new_formula[k], formula, fml_shape);
                new_formula[k][idx] = i;
                valid[k] = i;
                k++;
            }
            cudaMemcpy(d_valid, valid, 4*count, cudaMemcpyHostToDevice);

            if (formula[idx-1] < 2){
                temp_op_new = formula[idx-1];
                if (num_per_grp != 1){
                    cudaMalloc((void**)&new_temp_1, 8*count*rows);
                    copy_from_operands<<<num_block, 256>>>(new_temp_1, OPERAND, d_valid, rows, count);
                    cudaDeviceSynchronize();
                }
            } else {
                temp_op_new = temp_op;
                cudaMalloc((void**)&new_temp_1, 8*count*rows);
                if (formula[idx-1] == 2){
                    update_temp_weight<<<num_block, 256>>>(new_temp_1, temp_1, OPERAND, d_valid, rows, count, true);
                    cudaDeviceSynchronize();
                } else {
                    update_temp_weight<<<num_block, 256>>>(new_temp_1, temp_1, OPERAND, d_valid, rows, count, false);
                    cudaDeviceSynchronize();
                }
            }

            for (i=0; i<groups; i++){
                if (idx+2 == f_struct[i][2]){
                    chk = true;
                    break;
                }
            }
            if (chk || idx+1 == fml_shape){
                temp_0_change = true;
                cudaMalloc((void**)&new_temp_0, 8*count*rows);
                if (!temp_op_new){
                    if (num_per_grp != 1){
                        update_last_weight<<<num_block, 256>>>(new_temp_0, temp_0, new_temp_1, rows, count, true);
                        cudaDeviceSynchronize();
                    } else {
                        update_last_weight_through_operands<<<num_block, 256>>>(
                            new_temp_0, temp_0, OPERAND, d_valid, rows, count, true
                        );
                        cudaDeviceSynchronize();
                    }
                } else {
                    if (num_per_grp != 1){
                        update_last_weight<<<num_block, 256>>>(new_temp_0, temp_0, new_temp_1, rows, count, false);
                        cudaDeviceSynchronize();
                    } else {
                        update_last_weight_through_operands<<<num_block, 256>>>(
                            new_temp_0, temp_0, OPERAND, d_valid, rows, count, false
                        );
                        cudaDeviceSynchronize();
                    }
                }
            }
            else temp_0_change = false;

            if (idx+1 != fml_shape){
                if (chk){
                    if (add_sub){
                        new_idx = idx + 2;
                        new_mode = 1;
                    } else {
                        new_idx = idx + 1;
                        new_mode = 0;
                    }
                } else {
                    if (mul_div){
                        new_idx = idx + 2;
                        new_mode = 1;
                    } else {
                        new_idx = idx + 1;
                        new_mode = 2;
                    }
                }

                if (temp_0_change){
                    if (num_per_grp != 1){
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            new_temp_0+i*rows, temp_op_new, new_temp_1+i*rows,
                                            new_mode, add_sub, mul_div)) return true;
                    } else {
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            new_temp_0+i*rows, temp_op_new, temp_1,
                                            new_mode, add_sub, mul_div)) return true;
                    }
                } else {
                    if (num_per_grp != 1){
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            temp_0, temp_op_new, new_temp_1+i*rows,
                                            new_mode, add_sub, mul_div)) return true;
                    } else {
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            temp_0, temp_op_new, temp_1,
                                            new_mode, add_sub, mul_div)) return true;
                    }
                }
            }
            else {
                cudaError_t status = cudaMemcpy(
                    temp_weight_storage+count_temp_storage*rows,
                    new_temp_0, 8*count*rows, cudaMemcpyDeviceToDevice
                );
                if (status){
                    raise_error("Cuda bad status", "cudaMemcpy");
                }
                for (i=0; i<count; i++){
                    memcpy(temp_formula_storage[count_temp_storage+i], new_formula[i], fml_shape);
                }
                count_temp_storage += count;
                for (i=0; i<fml_shape; i++) current[0][i] = formula[i];
                current[0][fml_shape-1] = cols;
                if (count_temp_storage >= config.storage_size){
                    replace_nan_and_inf<<<count_temp_storage*rows/256 + 1, 256>>>(
                        temp_weight_storage, rows, count_temp_storage
                    );
                    cudaDeviceSynchronize();
                    if (compute_result(false)) return true;
                }
            }

            // Giai phong bo nho
            for (i=0; i<count; i++){
                delete[] new_formula[i];
            }
            delete[] new_formula;
            delete[] valid;
            cudaFree(d_valid);
            if (temp_0_change) cudaFree(new_temp_0);
            if (num_per_grp != 1) cudaFree(new_temp_1);
        }

        // Giai phong bo nho
        delete[] valid_operand;
    }
    return false;
}


bool Generator::compute_result(bool force_save){
    raise_error("Ham khong chay duoc", "Generator::compute_result");
    return true;
}


void Generator::run(){
    bool first = true;
    int i;
    uint8_t *formula;
    int **f_struct;
    double *temp_0, *temp_1;
    cudaMalloc((void**)&temp_0, 8*rows);
    cudaMalloc((void**)&temp_1, 8*rows);

    double *h_temp_0 = new double[rows];
    for (i=0; i<rows; i++) h_temp_0[i] = 0.0;

    string command;
    command.reserve(200);

    // Loop num_opr_per_fml
    while (true){
        command = "python " + config.lib_abs_path + "/PyScripts/create_table_PolyMethod.py "
            + config.folder_save + "/f.db 0 " + to_string(config.num_cycle-1) + " "
            + to_string(num_opr_per_fml) + " ";
        for (i=0; i<config.filter_field.size(); i++) command += config.filter_field[i] + " ";
        command.pop_back();
        system(command.c_str());
        fml_shape = num_opr_per_fml * 2;

        // Khoi tao formula
        formula = new uint8_t[fml_shape];
        for (i=0; i<config.storage_size+cols; i++)
            temp_formula_storage[i] = new uint8_t[fml_shape];
        
        for (i=0; i<config.num_cycle; i++){
            queries[i] = init_insert_query(num_opr_per_fml, i);
            queries[i].reserve(1000000);
        }

        // Loop num_opr_per_grp
        for (num_per_grp=1; num_per_grp<=num_opr_per_fml; num_per_grp++){
            if (num_opr_per_fml%num_per_grp || num_per_grp < current[1][0]) continue;
            groups = num_opr_per_fml / num_per_grp;
            f_struct = new int*[groups];
            for (i=0; i<groups; i++){
                f_struct[i] = new int[4];
                f_struct[i][0] = 0;
                f_struct[i][1] = num_per_grp;
                f_struct[i][2] = 1 + 2*num_per_grp*i;
                f_struct[i][3] = 0;
            }
            current[1][0] = num_per_grp;
            for (i=0; i<2*num_opr_per_fml; i++) formula[i] = 0;

            cudaMemcpy(temp_0, h_temp_0, 8*rows, cudaMemcpyHostToDevice);
            cudaMemcpy(temp_1, h_temp_0, 8*rows, cudaMemcpyHostToDevice);

            if (first) first = false;
            else {
                for (i=0; i<fml_shape; i++) current[0][i] = 0;
            }
            if (fill_formula(formula, f_struct, 0, temp_0, 0, temp_1, 0, false, false)) return;
            replace_nan_and_inf<<<count_temp_storage*rows/256 + 1, 256>>>(
                temp_weight_storage, rows, count_temp_storage
            );
            cudaDeviceSynchronize();
            if (compute_result(true)) return;

            //
            for (i=0; i<groups; i++) delete[] f_struct[i];
            delete[] f_struct;
        }

        //
        delete[] formula;
        for (i=0; i<config.storage_size+cols; i++) delete[] temp_formula_storage[i];

        //
        num_opr_per_fml += 1;
        current[2][0] = 0;
        current[1][0] = 1;
        delete[] current[0];
        current[0] = new int64_t[2*num_opr_per_fml];
        for (i=0; i<2*num_opr_per_fml; i++)
            current[0][i] = 0;
        
        command = "python " + config.lib_abs_path + "/PyScripts/create_checkpoint_PolyMethod.py "
            + config.folder_save + "/f.db " + to_string(num_opr_per_fml);
        system(command.c_str());
    }

    //
    delete[] h_temp_0;
    cudaFree(temp_0);
    cudaFree(temp_1);
}
