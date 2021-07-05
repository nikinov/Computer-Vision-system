using System;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Wision.CppRoutinesWrapper;

namespace ModelMaker
{
    public class Predictor
    {
        private static byte[] ConvertBmpToJpeg(byte[] bmp)
        {
            using (System.Drawing.Image image = System.Drawing.Image.FromStream(new MemoryStream(bmp)))
            {
                MemoryStream ms = new MemoryStream();
                image.Save(ms, ImageFormat.Jpeg);
                return ms.ToArray();
            }
        }
        //public const string Library = @"Predictor_dll.dll";
        //public const string Library = @"..\..\..\..\resources\C++\PredictorDll\build\Debug\Predictor_Dll.dll";
        public const string Library = @"C:\Temp\Predictor\Predictor_Dll.dll";

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int DLL_test();
        public static int Test()
        { 
            return DLL_test();
        }

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int DLL_GetPrediction(byte[] modelPath, IntPtr imageData, int imHight, int imWidth);

        /// <summary>
        /// get a prediction in a form of an int
        /// </summary>
        /// <param name="modelPath"></param> path to the model
        /// <param name="imageData"></param> image data in the form of a byte array
        /// <param name="imHight"></param> image hight
        /// <param name="imWidth"></param> image width
        /// <returns>int representing the label </returns>
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