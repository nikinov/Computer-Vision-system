using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ModelMaker;
using System.Drawing;
using System.Runtime.InteropServices;

namespace Detector
{
    class Program
    {
        static byte[] ImageToByteArray(Image imageIn)
        {
            using (var ms = new MemoryStream())
            {
                imageIn.Save(ms, imageIn.RawFormat);
                return ms.ToArray();
            }
        }
        [DllImport(@"C:\Users\Ryzen7-EXT\Documents\Github\WickonHightech\resources\C++\PredictorDll\dllmain.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int test();
        [DllImport(@"C:\Users\Ryzen7-EXT\Documents\Github\WickonHightech\resources\C++\PredictorDll\dllmain.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int GetPrediction(string modelPath, byte[] image, int imHight, int imWidth);
        static void Main(string[] args)
        {
            Console.WriteLine(Path.Combine(Directory.GetCurrentDirectory(),"..\\..\\"));
            //Maker.MakeModel(use_config: true, save_config: true, outPath: "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/scr/ModelMaker");
            Image im = Image.FromFile("C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/Assets/_Bad_Brown/01054268421_testfield_1031_3.bmp");
            Console.WriteLine(GetPrediction("C:/Users/Ryzen7 - EXT/Documents/Github/WickonHightech/scr/ModelMaker/model.pt", ImageToByteArray(im), 300, 300));
            Console.WriteLine(test());
            Console.ReadKey();
        }

    }
}
