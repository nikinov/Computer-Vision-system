#pragma once

#ifdef PREDICTORLIBRARY_EXPORTS
#define PREDICTORLIBRARY_API __declspec(dllexport)
#else
#define PREDICTORLIBRARY_API __declspec(dllimport)
#endif

extern "C" PREDICTORLIBRARY_API float* GetPrediction(const char* modelPath, unsigned char* byteArray);
