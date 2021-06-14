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
    EXPORT int DLL_GetPrediction(const char* modelPath, unsigned char imageData[], int imHight, int imWidth);

#ifdef __cplusplus
}
#endif 
