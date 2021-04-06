using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Microsoft.ML.OnnxRuntime;

namespace TestProject1
{
    class Program
    {
        private static string[] testFilesJPG = new[] { "../../../Test/VoidTest1.jpg" };
        private static string[] testFilesBMP = new[] { "../../../Test/BrownTest1.bmp" };
        static void Main(string[] args)
        {
            string modelDir = "../../../model.onnx";

            /*
                Configuring the the ONNX model to work with ML.net.
                Fitting the model to work with ModelInput class.
            */
            var context = new MLContext();

            var emptyData = new List<ModelInput>();

            var data = context.Data.LoadFromEnumerable(emptyData);

            var pipeline = context.Transforms.ResizeImages(resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill,
                outputColumnName: "data", imageHeight: 300, imageWidth: 300,
                inputColumnName: nameof(ModelInput.image))
                .Append(context.Transforms.ExtractPixels(outputColumnName: "data"))
                .Append(context.Transforms.ApplyOnnxModel(modelFile: modelDir, outputColumnName: "model_output", inputColumnName: "data"));

            var model = pipeline.Fit(data);

            var predictionEngine = context.Model.CreatePredictionEngine<ModelInput, ModelPrediction>(model);

            // making Prediction on JPG
            foreach (var image in testFilesJPG)
            {
                // opening the JPG image
                Bitmap testImage;
                using (var stream = new FileStream(image, FileMode.Open))
                {
                    testImage = (Bitmap)Image.FromStream(stream);
                }

                // making a prediction
                var prediction = predictionEngine.Predict(new ModelInput { image = testImage });

                Console.WriteLine("---------------------------------------------------------------------");
                Console.WriteLine("I have this much confidence that it's a Brown Defect: " + prediction.PredictedLabels[0]);
                Console.WriteLine("I have this much confidence that it's a Void Defect: " + prediction.PredictedLabels[1]);
                Console.WriteLine("---------------------------------------------------------------------");
            }

            // making Prediction on BMP
            foreach (string image in testFilesBMP)
            {
                // opening the PMB image
                Bitmap testImage;
                using (var img = new FileStream(image, FileMode.Open))
                {
                    testImage = new Bitmap(img);
                }

                // making a prediction
                var prediction = predictionEngine.Predict(new ModelInput { image = testImage });

                Console.WriteLine("---------------------------------------------------------------------");
                Console.WriteLine("I have this much confidence that it's a Brown Defect: " + prediction.PredictedLabels[0]);
                Console.WriteLine("I have this much confidence that it's a Void Defect: " + prediction.PredictedLabels[1]);
                Console.WriteLine("---------------------------------------------------------------------");
            }
        }
    }
}
