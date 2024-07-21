#pragma once
#include <omp.h>
#include <cmath>
using namespace std;


const double __NEGATIVE_INFINITY__ = -1.7976931348623157e+308;
const double __POSITIVE_INFINITY__ = +1.7976931348623157e+308;


inline void cuda_set_array_value(double* array, int length, double value){
    #pragma omp parallel for
    for (int index=0; index<length; index++){
        array[index] = value;
    }
}


inline void copy_from_operands(double *dest, double *operands, int *arrCpy, int length, int numCpy){
    #pragma omp parallel for
    for (int index=0; index<numCpy*length; index++){
        int i = index / length;
        int j = index % length;
        dest[index] = operands[arrCpy[i]*length + j];
    }
}


inline void update_temp_weight(double *temp_weight_new, double *temp_weight_old, double *operands, int *arrOpr, int length, int numOpr, bool isMul){
    #pragma omp parallel for
    for (int index=0; index<numOpr*length; index++){
        int i = index / length;
        int j = index % length;
        if (isMul)
            temp_weight_new[index] = temp_weight_old[j] * operands[arrOpr[i]*length + j];
        else
            temp_weight_new[index] = temp_weight_old[j] / operands[arrOpr[i]*length + j];
    }
}


inline void update_last_weight(double *last_weight, double *curr_weight, double *temp_weight, int length, int numOpr, bool isAdd){
    #pragma omp parallel for
    for (int index=0; index<numOpr*length; index++){
        int j = index % length;
        if (isAdd) last_weight[index] = curr_weight[j] + temp_weight[index];
        else last_weight[index] = curr_weight[j] - temp_weight[index];
    }
}


inline void update_last_weight_through_operands(double *last_weight, double *curr_weight, double *operands, int *arrOpr, int length, int numOpr, bool isAdd){
    #pragma omp parallel for
    for (int index=0; index<numOpr*length; index++){
        int i = index / length;
        int j = index % length;
        if (isAdd) last_weight[index] = curr_weight[j] + operands[arrOpr[i]*length + j];
        else last_weight[index] = curr_weight[j] - operands[arrOpr[i]*length + j];
    }
}


inline void replace_nan_and_inf(double *array, int length, int numOpr){
    #pragma omp parallel for
    for (int index=0; index<numOpr*length; index++){
        if (isnan(array[index]) || isinf(array[index]))
            array[index] = __NEGATIVE_INFINITY__;
    }
}
