using System;
using System.IO;
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
            //ModelMaker.Predictor.Test();

            

            //Console.WriteLine(Path.Combine(Directory.GetCurrentDirectory(),"..\\..\\"));
            //Maker.MakeModel(use_config: true, save_config: true, outPath: "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/scr/ModelMaker");
            Image im = Image.FromFile("C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/python/otherImages/_Bad_Brown/01056584880_testfield_1031_2.bmp");
            Console.WriteLine(ModelMaker.Predictor.GetPrediction("C:/Users/Ryzen7 - EXT/Documents/Github/WickonHightech/resources/models/model.pt", ImageToByteArray(im), 300, 300));
            //Console.WriteLine(test());
            Console.ReadKey();
        }
    }
}
