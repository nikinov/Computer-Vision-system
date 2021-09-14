#pragma once


#if defined(_MSC_VER)
//  Microsoft 
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)

//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

/*
extern "C" {

    // return DLL version info
    EXPORT int DLL_GetPrediction(
        const char* modelPath,
        unsigned char* byteArray,
        float* buffer,                      // output char array with version info message
        int allocSiz                        // size of externally allocated array
    );


}
*/

using namespace std;

#ifdef __cplusplus
extern "C" {

#endif

    EXPORT int DLL_test();
    EXPORT void DLL_GetPrediction(unsigned char* imageData, int imHeight, int imWidth, float* out, int sizOut, bool resnet);
    EXPORT void DLL_InitModel(const char* modelPath);

#ifdef __cplusplus
}
#endif 
