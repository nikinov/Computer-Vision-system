@echo off

:: CREATE PROJECT
cmake -G "Visual Studio 16 2019" -A x64 -S . -B .\build
pause