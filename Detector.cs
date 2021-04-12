using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using System.Drawing;
using System.IO;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using System.Threading;
using System.Net;

namespace ONNXDetector
{
    public class Detector
    {
        private PredictionEngine<ModelInput, ModelPrediction> predictionEngine = null;
        private string[] labels = null;

        /*
         custom vision objects
         */
        // You can obtain these values from the Keys and Endpoint page for your Custom Vision resource in the Azure Portal.
        public string trainingEndpoint = "https://westeurope.api.cognitive.microsoft.com/";
        public string trainingKey = "44a1cdd0f2194aabbb23b158752bf9eb";
        // You can obtain this value from the Properties page for your Custom Vision Prediction resource in the Azure Portal. See the "Resource ID" field. This typically has a value such as:
        // /subscriptions/<your subscription ID>/resourceGroups/<your resource group>/providers/Microsoft.CognitiveServices/accounts/<your Custom Vision prediction resource name>
        public string predictionResourceId = "/subscriptions/70bfdebf-70f6-44e5-829b-4cb8e034b648/resourceGroups/Wickon/providers/Microsoft.CognitiveServices/accounts/WickonDetector";

        private static List<ImageSet> imagePaths = new List<ImageSet>();
        private static List<Tag> labelTags = new List<Tag>();
        private static Iteration iteration;
        CustomVisionTrainingClient trainingApi = null;
        /*
         custom vision objects
         */

        public void Init(string modelDir, string labelsDir)
        {
            /*
             custom vision calls
             */

            /*
             custom vision calls
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

        public void CreateAndTrainModel(string projectName, string trainingAssetDir)
        {
            trainingApi = AuthenticateTraining(trainingEndpoint, trainingKey);
            Project project = CreateProject(trainingApi, projectName);
            AddTags(trainingApi, project, trainingAssetDir);
            LoadImagesFromDisk(labelTags, trainingAssetDir);
            foreach(Tag tag in labelTags)
            {
                UploadImages(trainingApi, project, tag);
            }
            TrainProject(trainingApi, project);
            PublishIteration(trainingApi, project, projectName);
            Export export = new Export(platform: "ONNX");
            while (export.Status == "Exporting")
            {
                Console.WriteLine("Exporting...");
                Thread.Sleep(1000);
            }
            if (export.Status == "Done")
            {
                 using (var client = new WebClient())
                 {
                    client.DownloadFile(export.DownloadUri, "model");
                 }
            }
            else if (export.Status == "Failed")
            {
                throw new Exception("the model was not exported");
            }
        }

        private static CustomVisionTrainingClient AuthenticateTraining(string endpoint, string trainingKey)
        {
            // Create the Api, passing in the training key
            CustomVisionTrainingClient trainingApi = new CustomVisionTrainingClient(new Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.ApiKeyServiceClientCredentials(trainingKey))
            {
                Endpoint = endpoint
            };
            return trainingApi;
        }

        private static Project CreateProject(CustomVisionTrainingClient trainingApi, string projectName)
        {
            // Create a new project
            Console.WriteLine("Creating new project:");
            return trainingApi.CreateProject(projectName);
        }

        private static void AddTags(CustomVisionTrainingClient trainingApi, Project project, string path)
        {
            var directories = Directory.GetDirectories(path);
            // Make two tags in the new project
            foreach (string tag in directories)
            {
                labelTags.Add(trainingApi.CreateTag(project.Id, tag));
            }
        }

        private static void LoadImagesFromDisk(List<Tag> labels, string assetPath)
        {
            // this loads the images to be uploaded from disk into memory
            foreach(Tag label in labels)
            {
                imagePaths.Add(new ImageSet() { imagePaths = Directory.GetFiles(Path.Combine(assetPath, label.Name)).ToList() });
            }
        }

        private static void UploadImages(CustomVisionTrainingClient trainingApi, Project project, Tag tag)
        {
            // Images can be uploaded one at a time
            foreach (var image in imagePaths[labelTags.IndexOf(tag)].imagePaths)
            {
                using (var stream = new MemoryStream(File.ReadAllBytes(image)))
                {
                    trainingApi.CreateImagesFromData(project.Id, stream, new List<Guid>() { tag.Id });
                }
            }

        }

        private static void TrainProject(CustomVisionTrainingClient trainingApi, Project project)
        {
            // Now there are images with tags start training the project
            Console.WriteLine("\tTraining");
            iteration = trainingApi.TrainProject(project.Id);

            // The returned iteration will be in progress, and can be queried periodically to see when it has completed
            while (iteration.Status == "Training")
            {
                Console.WriteLine("iterating...");
                Thread.Sleep(10000);

                // Re-query the iteration to get it's updated status
                iteration = trainingApi.GetIteration(project.Id, iteration.Id);
            }
        }
        private void PublishIteration(CustomVisionTrainingClient trainingApi, Project project, string projectName)
        {
            trainingApi.PublishIteration(project.Id, iteration.Id, "M" + projectName, predictionResourceId);
            Console.WriteLine("Done!\n");

            // Now there is a trained endpoint, it can be used to make a prediction
        }
    }
    class ImageSet
    {
        public List<string> imagePaths;
    }
}
