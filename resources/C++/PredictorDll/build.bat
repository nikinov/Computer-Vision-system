g++ -c -DBUILD_MY_DLL dllmain.cpp
g++ -shared -o dllmain.dll dllmain.o -Wl,--out-implib,libdllmain.a