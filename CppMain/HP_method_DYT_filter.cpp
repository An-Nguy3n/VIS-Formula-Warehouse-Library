#include "../CppSources/Filter/DoubleYearThreshold/DoubleYearThresholdFilter.cpp"
#include "omp.h"
#include <iostream>
using namespace std;


int main(int argc, char *argv[]){
    if (argc == 1){
        raise_error("Command chua nhap config_path", ".\\ExeFile\\HP_method_DYT_filter_CPU.exe <config_path>");
    }

    int num_threads = omp_get_max_threads()-2;
    cout << "Num threads: " << num_threads << endl;
    omp_set_num_threads(num_threads);
    string config_path = argv[1];
    DoubleYearThresholdFilter vis(config_path);
    vis.run();
}
