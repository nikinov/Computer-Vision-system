using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using System.Drawing;
using System.IO;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using System.Threading;

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
        private static string trainingEndpoint = "https://westeurope.api.cognitive.microsoft.com/";
        private static string trainingKey = "44a1cdd0f2194aabbb23b158752bf9eb";
        // You can obtain these values from the Keys and Endpoint page for your Custom Vision Prediction resource in the Azure Portal.
        private static string predictionEndpoint = "<your prediction endpoint here>";
        private static string predictionKey = "<your prediction key here>";
        // You can obtain this value from the Properties page for your Custom Vision Prediction resource in the Azure Portal. See the "Resource ID" field. This typically has a value such as:
        // /subscriptions/<your subscription ID>/resourceGroups/<your resource group>/providers/Microsoft.CognitiveServices/accounts/<your Custom Vision prediction resource name>
        private static string predictionResourceId = "/subscriptions/70bfdebf-70f6-44e5-829b-4cb8e034b648/resourceGroups/Wickon/providers/Microsoft.CognitiveServices/accounts/WickonDetector";

        private static List<string> hemlockImages;
        private static List<string> japaneseCherryImages;
        private static Tag hemlockTag;
        private static Tag japaneseCherryTag;
        private static Iteration iteration;
        private static string publishedModelName = "treeClassModel";
        private static MemoryStream testImage;
        /*
         custom vision objects
         */

        public void Init(string modelDir, string labelsDir)
        {
            /*
             custom vision calls
             */
            CustomVisionTrainingClient trainingApi = AuthenticateTraining(trainingEndpoint, trainingKey);
            CustomVisionPredictionClient predictionApi = AuthenticatePrediction(predictionEndpoint, predictionKey);

            Project project = CreateProject(trainingApi);
            AddTags(trainingApi, project);
            UploadImages(trainingApi, project);
            TrainProject(trainingApi, project);
            PublishIteration(trainingApi, project);
            TestIteration(predictionApi, project);
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

        private static CustomVisionTrainingClient AuthenticateTraining(string endpoint, string trainingKey)
        {
            // Create the Api, passing in the training key
            CustomVisionTrainingClient trainingApi = new CustomVisionTrainingClient(new Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.ApiKeyServiceClientCredentials(trainingKey))
            {
                Endpoint = endpoint
            };
            return trainingApi;
        }
        private static CustomVisionPredictionClient AuthenticatePrediction(string endpoint, string predictionKey)
        {
            // Create a prediction endpoint, passing in the obtained prediction key
            CustomVisionPredictionClient predictionApi = new CustomVisionPredictionClient(new Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction.ApiKeyServiceClientCredentials(predictionKey))
            {
                Endpoint = endpoint
            };
            return predictionApi;
        }

        private static Project CreateProject(CustomVisionTrainingClient trainingApi)
        {
            // Create a new project
            Console.WriteLine("Creating new project:");
            return trainingApi.CreateProject("My New Project");
        }

        private static void AddTags(CustomVisionTrainingClient trainingApi, Project project)
        {
            // Make two tags in the new project
            hemlockTag = trainingApi.CreateTag(project.Id, "Hemlock");
            japaneseCherryTag = trainingApi.CreateTag(project.Id, "Japanese Cherry");
        }

        private static void LoadImagesFromDisk()
        {
            // this loads the images to be uploaded from disk into memory
            hemlockImages = Directory.GetFiles(Path.Combine("Images", "Hemlock")).ToList();
            japaneseCherryImages = Directory.GetFiles(Path.Combine("Images", "Japanese Cherry")).ToList();
            testImage = new MemoryStream(File.ReadAllBytes(Path.Combine("Images", "Test\\test_image.jpg")));
        }

        private static void UploadImages(CustomVisionTrainingClient trainingApi, Project project)
        {
            // Add some images to the tags
            Console.WriteLine("\tUploading images");
            LoadImagesFromDisk();

            // Images can be uploaded one at a time
            foreach (var image in hemlockImages)
            {
                using (var stream = new MemoryStream(File.ReadAllBytes(image)))
                {
                    trainingApi.CreateImagesFromData(project.Id, stream, new List<Guid>() { hemlockTag.Id });
                }
            }

            // Or uploaded in a single batch 
            var imageFiles = japaneseCherryImages.Select(img => new ImageFileCreateEntry(Path.GetFileName(img), File.ReadAllBytes(img))).ToList();
            trainingApi.CreateImagesFromFiles(project.Id, new ImageFileCreateBatch(imageFiles, new List<Guid>() { japaneseCherryTag.Id }));

        }

        private static void TrainProject(CustomVisionTrainingClient trainingApi, Project project)
        {
            // Now there are images with tags start training the project
            Console.WriteLine("\tTraining");
            iteration = trainingApi.TrainProject(project.Id);

            // The returned iteration will be in progress, and can be queried periodically to see when it has completed
            while (iteration.Status == "Training")
            {
                Console.WriteLine("Waiting 10 seconds for training to complete...");
                Thread.Sleep(10000);

                // Re-query the iteration to get it's updated status
                iteration = trainingApi.GetIteration(project.Id, iteration.Id);
            }
        }
        private static void PublishIteration(CustomVisionTrainingClient trainingApi, Project project)
        {
            trainingApi.PublishIteration(project.Id, iteration.Id, publishedModelName, predictionResourceId);
            Console.WriteLine("Done!\n");

            // Now there is a trained endpoint, it can be used to make a prediction
        }

        private static void TestIteration(CustomVisionPredictionClient predictionApi, Project project)
        {
            // Make a prediction against the new project
            Console.WriteLine("Making a prediction:");
            var result = predictionApi.ClassifyImage(project.Id, publishedModelName, testImage);

            // Loop over each prediction and write out the results
            foreach (var c in result.Predictions)
            {
                Console.WriteLine($"\t{c.TagName}: {c.Probability:P1}");
            }
        }


    }
}
