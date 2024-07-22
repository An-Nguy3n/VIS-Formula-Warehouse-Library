#include "CppSources/Filter/SingleYearThreshold/SingleYearThresholdFilter.cu"


int main(int argc, char *argv[]){
    if (argc == 1){
        raise_error("Command chua nhap config_path", ".\\ExeFile\\HP_method_SYT_filter_CUDA.exe <config_path>");
    }

    string config_path = argv[1];
    SingleYearThresholdFilter vis(config_path);
    vis.run();
}
