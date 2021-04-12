# WickonHightech

## predict on a ONNX model
Optical inspections systems.

### about
it's a simple demo that makes prediction on an ONNX model
 - you give it an ONNX model which is a neural network file type
 - then you give it the image you want to detect
 - it returns the type of defect that the model thinks the circuit from the image has
 - ONNX detection model.

### How to Use

add the libuarry ONNXDetector
```C#
using ONNXDetector;
```

initialize the detector in main and call init
```C#
Detector detector = new Detector();
detector.Init("../../../model.onnx", "../../../Labels.txt");
```

to predict an image on the onnx model run one of the following
```C#
detector.GetPrediction(bitmapImage) // returns a float array
detector.GetPredictionLabel(bitmapImage) // returns the predicted string label
```

## creating and training a model

### about

 - this is a quick demo on how to use the Custom Vision Rest api to train and get an ONNX mdoel

### how to use
you can call CreateAndTrain, it will make a folder with an ONNX model and a labels file that you can then use in the Prediction,
you can optionally specify the trainingEndpoint, trainingKey, predictionResourceId for your own account

```C#
// specifying the optional parameters
// You can obtain these values from the Keys and Endpoint page for your Custom Vision resource in the Azure Portal.
detector.trainingEndpoint = "https://westeurope.api.cognitive.microsoft.com/";
detector.trainingKey = "44a1cdd0f2194aabbb23b158752bf9eb";
// You can obtain this value from the Properties page for your Custom Vision Prediction resource in the Azure Portal. See the "Resource ID" field. This typically has a value such as:
// /subscriptions/<your subscription ID>/resourceGroups/<your resource group>/providers/Microsoft.CognitiveServices/accounts/<your Custom Vision prediction resource name>
detector.predictionResourceId = "/subscriptions/70bfdebf-70f6-44e5-829b-4cb8e034b648/resourceGroups/Wickon/providers/Microsoft.CognitiveServices/accounts/WickonDetector";

// creating and training a model on my data
detector.CreateAndTrainModel("MyProjectName", "MyAssetDirectory");
```

## external links

- more informatinon about the ONNX model https://onnx.ai/
- info for the custom vision rest api https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/quickstarts/image-classification?tabs=visual-studio&pivots=programming-language-csharp
