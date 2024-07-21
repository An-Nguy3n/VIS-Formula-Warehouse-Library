#include "CppSources/Filter/DoubleYearThreshold/DoubleYearThresholdFilter.cpp"
#include <omp.h>


int main(int argc, char *argv[]){
    if (argc == 1){
        raise_error("Command chua nhap config_path", ".\\ExeFile\\HP_method_DYT_filter_CPU.exe <config_path>");
    }

    // omp_set_num_threads(8);
    string config_path = argv[1];
    DoubleYearThresholdFilter vis(config_path);
    vis.run();
}
