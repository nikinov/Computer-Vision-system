using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using System.Drawing;
using System.IO;

namespace ONNXDetector
{
    public class Detector
    {
        private PredictionEngine<ModelInput, ModelPrediction> predictionEngine = null;
        private string[] labels = null;

        public void Init(string modelDir, string labelsDir)
        {
            var context = new MLContext();

            var emptyData = new List<ModelInput>();

            var data = context.Data.LoadFromEnumerable(emptyData);

            var pipeline = context.Transforms.ResizeImages(resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill,
                outputColumnName: "data", imageHeight: 300, imageWidth: 300,
                inputColumnName: nameof(ModelInput.image))
                .Append(context.Transforms.ExtractPixels(outputColumnName: "data"))
                .Append(context.Transforms.ApplyOnnxModel(modelFile: modelDir, outputColumnName: "model_output", inputColumnName: "data"));

            var model = pipeline.Fit(data);

            predictionEngine = context.Model.CreatePredictionEngine<ModelInput, ModelPrediction>(model);

            labels = File.ReadAllLines(labelsDir);
        }

        public float[] GetPrediction(Bitmap bitmapImage)
        {
            try
            {
                var prediction = predictionEngine.Predict(new ModelInput { image = bitmapImage });
                return prediction.PredictedLabels;
            }
            catch (Exception)
            {
                Console.WriteLine("ERROR: You have probably forgot to call Detector.Init(modelDir) at the start of the program!");
                throw;
            }
            return new float[0];
        }

        public string GetPredictionLabel(Bitmap bitmapImage)
        {
            try
            {
                var prediction = predictionEngine.Predict(new ModelInput { image = bitmapImage });

                return labels[prediction.PredictedLabels.ToList().IndexOf(prediction.PredictedLabels.Max())];

            }
            catch (Exception)
            {
                Console.WriteLine("ERROR: You have probably forgot to call Detector.Init(modelDir) at the start of the program!");
                throw;
            }
            return "None";
        }
    }
}
