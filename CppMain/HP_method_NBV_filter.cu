#include "../CppSources/Filter/NoBalanceValue/NoBalanceValueFilter.cu"


int main(int argc, char *argv[]){
    if (argc == 1){
        raise_error("Command chua nhap config_path", ".\\ExeFile\\HP_method_NBV_filter_CUDA.exe <config_path>");
    }

    string config_path = argv[1];
    NoBalanceValueFilter vis(config_path);
    vis.run();
}
