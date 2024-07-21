#include "CppSources/Filter/DoubleYearThreshold/DoubleYearThresholdFilter.cu"


int main(int argc, char *argv[]){
    string config_path = argv[1];
    DoubleYearThresholdFilter vis(config_path);
    vis.run();
}
