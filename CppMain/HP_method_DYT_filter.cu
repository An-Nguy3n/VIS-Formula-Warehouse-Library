#include "../CppSources/Filter/DoubleYearThreshold/DoubleYearThresholdFilter.cu"


int main(int argc, char *argv[]){
    if (argc == 1){
        raise_error("Command chua nhap config_path", ".\\ExeFile\\HP_method_DYT_filter_CUDA.exe <config_path>");
    }

    string config_path = argv[1];
    DoubleYearThresholdFilter vis(config_path);
    vis.run();
}
