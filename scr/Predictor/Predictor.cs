using System;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using RoutinesWrapper;
using System.Runtime.InteropServices;

namespace Predictor
{
    public class Predictor
    {
        public const string Library = @"C:\Users\Ryzen7-EXT\Documents\Github\WickonHightech\resources\C++\numPredictor\build\Debug\Predictor_Dll.dll";

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr DLL_GetPrediction(IntPtr imageData, int imHight, int imWidth, out IntPtr output, int outSize, bool resnet);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern void DLL_InitModel(byte[] modelPath);

        public static IntPtr GetPrediction(
        byte[] imageData,
        int imHight,
        int imWidth,
        int outSize,
        bool resnet)
        {
            IntPtr output;
            using (var pinImageData = imageData.Pin())
            {
                DLL_GetPrediction(pinImageData, imHight, imWidth, out output, outSize, resnet);
                return output;
            }
        }

        public static void InitModel(string modelPath)
        {
            var buffer = Encoding.ASCII.GetBytes(modelPath);
            DLL_InitModel(buffer);
        }
    }
}
