using System;
using System.IO;
using System.Drawing;
using System.Windows.Media.Imaging;
using ModelMaker;

namespace Detector
{
    class Program
    {
        static void Main(string[] args)
        {
            			if (false) // for model maker use true, or use "example.py" to create the "model.pt" !
			{
				//Console.WriteLine(Path.Combine(Directory.GetCurrentDirectory(),"..\\..\\"));
				//Maker.MakeModel(use_config: true, save_config: true, outPath: "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/scr/ModelMaker");
				Maker.MakeModel(use_config: true, save_config: true, outPath: @"..\ModelMaker");
			}
			else
			{
				// DLL calling test
				Predictor.Test();
				// Predictor.GetPrediction(); ....

				var currentDir = Directory.GetCurrentDirectory();
				var modelFile = Path.Combine(currentDir, @"..\..\..\..\resources\models\model.pt");
				if (!File.Exists(modelFile))
				{
					throw new FileNotFoundException(modelFile);
				}
				//Image im = Image.FromFile(@"C:/temp/0.png");
				var byteArray = new byte[300 * 300 * 4];
				//Console.WriteLine(byteArray[0] + "\n" + byteArray[1] + "\n" + byteArray[2]);
				Console.WriteLine(ModelMaker.Predictor.GetPrediction(modelFile, byteArray, 300, 300));
				//Console.WriteLine(test());
			}

        }
    }
}
