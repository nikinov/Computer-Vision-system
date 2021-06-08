using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ModelMaker;
using System.Drawing;

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
        static void Main(string[] args)
        {
            Console.WriteLine(Path.Combine(Directory.GetCurrentDirectory(),"..\\..\\"));
            //Maker.MakeModel(use_config: true, save_config: true, outPath: "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/scr/ModelMaker");
            Image im = Image.FromFile("C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/Assets/_Bad_Brown/01054268421_testfield_1031_3.bmp");
            Console.WriteLine(Predictor.GetPrediction("C:/Users/Ryzen7 - EXT/Documents/Github/WickonHightech/scr/ModelMaker/model.pt", ImageToByteArray(im), 2));
            Console.ReadKey();
        }

    }
}
