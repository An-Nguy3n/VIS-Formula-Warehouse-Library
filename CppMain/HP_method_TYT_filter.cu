#include "../CppSources/Filter/TripleYearThreshold/TripleYearThresholdFilter.cu"


int main(int argc, char *argv[]){
    if (argc == 1){
        raise_error("Command chua nhap config_path", ".\\ExeFile\\HP_method_TYT_filter_CUDA.exe <config_path>");
    }

    string config_path = argv[1];
    TripleYearThresholdFilter vis(config_path);
    vis.run();
}
