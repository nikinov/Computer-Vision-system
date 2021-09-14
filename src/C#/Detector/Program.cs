using System;
using System.IO;
using System.Drawing;
using System.Windows.Media.Imaging;
using ModelMaker;
using Predictor;

namespace Detector
{
    class Program
    {
        static void Main(string[] args)
        {
            
            //Predictor.Test();
            //Console.WriteLine(Path.Combine(Directory.GetCurrentDirectory(),"..\\..\\"));
            //Maker.MakeModel(use_config: true, save_config: true, outPath: "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/scr/ModelMaker");
            //Image im = Image.FromFile(@"C:\Users\Ryzen7-EXT\Documents\C++Stuff\TorchTest\0.png");
            var test = new byte [300* 300* 3];
            Predictor.Predictor.InitModel("C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/python/new_vision_system/models/my_other_number_model.pt");
            Console.WriteLine(Predictor.Predictor.GetPrediction(test, 300, 300, 11, true));
            //Console.WriteLine(ImageToByteArray(im)[0] + "\n" + ImageToByteArray(im)[1] + "\n" + ImageToByteArray(im)[2]);
            //Console.WriteLine(ModelMaker.Predictor.GetPrediction("C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/models/model.pt", ImageToByteArray(im), 300, 300));
            //Console.WriteLine(test());
            Console.ReadKey();
        }
    }
}
