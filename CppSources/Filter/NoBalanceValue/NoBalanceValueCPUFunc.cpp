#pragma once
#include <omp.h>



inline void merge(
    double *keys,
    double *vals,
    int left,
    int mid,
    int right,
    double *temp_keys,
    double *temp_vals
) {
    int i = left, j = mid + 1;
    double nextKey, nextVal;
    int k = 0;

    while (i <= mid && j <= right){
        if (keys[i] >= keys[j]){
            nextKey = keys[i];
            nextVal = vals[i];
            i++;
        }
        else {
            nextKey = keys[j];
            nextVal = vals[j];
            j++;
        }
        temp_keys[left+k] = nextKey;
        temp_vals[left+k] = nextVal;
        k++;
    }

    while (i <= mid){
        temp_keys[left+k] = keys[i];
        temp_vals[left+k] = vals[i];
        i++; k++;
    }

    while (j <= right){
        temp_keys[left+k] = keys[j];
        temp_vals[left+k] = vals[j];
        j++; k++;
    }

    for (i=left; i<=right; i++){
        keys[i] = temp_keys[i];
        vals[i] = temp_vals[i];
    }
}


inline void mergeSort(
    double *keys,
    double *vals,
    int n,
    double *temp_keys,
    double *temp_vals
) {
    int curr_size;
    int left, mid, right;

    for (curr_size=1; curr_size<n; curr_size*=2){
        for (left=0; left<n-1; left+=2*curr_size){
            if (left+curr_size-1 < n-1) mid = left + curr_size - 1;
            else mid = n - 1;

            if (left+2*curr_size-1 < n-1) right = left + 2*curr_size - 1;
            else right = n - 1;

            merge(keys, vals, left, mid, right, temp_keys, temp_vals);
        }
    }
}


inline double mean(
    double *array,
    int start,
    int end
) {
    double temp = 0.0;
    for (int i=start; i<end; i++) temp += array[i];

    return temp / (end - start);
}


inline double noBalance_point(
    double *sortedProfit,
    int n
) {
    int cur_idx = n / 2;
    int i;

    for (i=cur_idx; i>0; i--){
        if (mean(sortedProfit, 0, i) <= mean(sortedProfit, i, n)){
            if (i == cur_idx) return 100000000.0;
            return i + 1.0;
        }
    }

    return 1.0;
}


inline void _get_noBalance_value(
    double *weight,
    double *profit,
    double *temp_wgt,
    double *temp_prf,
    int *INDEX,
    int index_size,
    int num_cycle,
    double *result
) {
    int i, start, end, n, rs_idx;
    double temp_sum = 0.0;
    int count = 0;

    for (i=index_size-2; i>0; i--){
        start = INDEX[i];
        end = INDEX[i+1];
        n = end - start;

        mergeSort(weight + start, profit + start, n, temp_wgt + start, temp_prf + start);

        temp_sum += noBalance_point(profit + start, n);
        count++;

        if (i <= num_cycle){
            rs_idx = num_cycle - i;
            if (temp_sum >= 1e8) result[rs_idx] = -1e8;
            else result[rs_idx] = -temp_sum / count;
        }
    }
}


void get_noBalance_value(
    double *weights,
    double *profits,
    double *temp_wgts,
    double *temp_prfs,
    int length,
    int *INDEX,
    int index_size,
    int num_cycle,
    double *results,
    int num_array
) {
    #pragma omp parallel for
    for (int index=0; index<num_array; index++){
        _get_noBalance_value(
            weights + index*length,
            profits + index*length,
            temp_wgts + index*length,
            temp_prfs + index*length,
            INDEX,
            index_size,
            num_cycle,
            results + index*num_cycle
        );
    }
}
