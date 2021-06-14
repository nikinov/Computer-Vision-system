using System;
using System.Runtime.InteropServices;
using System.Text;
using Wision.CppRoutinesWrapper;

namespace ModelMaker
{
    public class Predictor
    {
        //public const string Library = @"Predictor_dll.dll";
        public const string Library = @"c:\Users\Ryzen7-EXT\Documents\Github\WickonHightech\resources\C++\PredictorDll\Debug\Predictor_dll.dll";

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int DLL_test();
        public static int Test()
        { 
            return DLL_test();
        }

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int DLL_GetPrediction(byte[] modelPath, IntPtr imageData, int imHight, int imWidth);
        public static int GetPrediction(
            string modelPath,
            byte[] imageData,
            int imHight,
            int imWidth
        )
        {
            int status;
            var buffer = Encoding.ASCII.GetBytes(modelPath);
            using (var pinImageData = imageData.Pin())
            {
                status = DLL_GetPrediction(buffer, pinImageData, imHight, imWidth);
            }
            
            return status;
        }
    }

}