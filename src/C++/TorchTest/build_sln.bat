@echo off

:: ADD PATH
mkdir build
cd build
cmake -S ../ -B . -DCMAKE_PREFIX_PATH="C:/libtorch"

:: cmake --build C:\Users\Ryzen7-EXT\Documents\Github\WickonHightech\resources\C++\PredictorDll\build --target clean
:: CREATE PROJECT
:: cmake -G "Visual Studio 16 2019" -A x64 -S . -B .\build
pause
