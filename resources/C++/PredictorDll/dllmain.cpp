// dllmain.cpp : Defines the entry point for the DLL application.
//#include <cstring>
//#include <Windows.h>
#include "../Predictor/headers/Predictor.h"
#include "dllmain.h"

int GetPrediction(const char* modelPath, unsigned char imageData[], int imHight, int imWidth);
int test();

/*
// defined in Windows.h
#undef max
#undef min

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

int DLL_GetPrediction(
    const char* modelPath,
    unsigned char* byteArray,
    float* buffer,                      // output char array with version info message
    int allocSizOfBuffer                        // size of externally allocated array
)
{
    auto output = GetPrediction(modelPath, byteArray);

    std::memcpy(buffer, output.data<float>(), output.size(0));


    return 0;
}
*/