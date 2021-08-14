using System;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using RoutinesWrapper;

namespace Predictor
{
    public class Predictor
    {
        public const string Library = @"C:\Users\Ryzen7-EXT\Documents\Github\WickonHightech\resources\C++\numPredictor\build\Debug\Predictor_Dll.dll";


        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr DLL_GetPrediction(IntPtr imageData, int imHight, int imWidth);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern void DLL_InitModel(byte[] modelPath);

        public static IntPtr GetPrediction(
        byte[] imageData,
        int imHight,
        int imWidth)
        {
            using (var pinImageData = imageData.Pin())
            {
                return DLL_GetPrediction(pinImageData, imHight, imWidth);
            }
        }

        public static void InitModel(string modelPath)
        {
            var buffer = Encoding.ASCII.GetBytes(modelPath);
            DLL_InitModel(buffer);
        }
    }
}
