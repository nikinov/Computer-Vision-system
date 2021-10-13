@echo off

:: ADD PATH
mkdir build
cd build
cmake -S ../ -B . -DCMAKE_PREFIX_PATH="C:\vcpkg\installed\x64-windows"

:: CREATE PROJECT
:: cmake -G "Visual Studio 16 2019" -A x64 -S . -B .\build
pause
