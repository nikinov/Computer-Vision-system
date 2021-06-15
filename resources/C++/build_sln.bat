@echo off

:: ADD PATH
cmake -DCMAKE_PREFIX_PATH=C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/C++/Predictor/lib/libtorch

:: CREATE PROJECT
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 -S . -B .\build
pause