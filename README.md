# WickonHightech

Optical inspections systems.
 - ONNX detection model.

# How to Use

add the libuarry ONNXDetector
 - using ONNXDetector;

initialize the detector in main and call init
 - Detector detector = new Detector();
 - detector.Init("../../../model.onnx", "../../../Labels.txt");

to predict an image on the onnx model run one of the following
 - detector.GetPrediction(bitmapImage) returns a float array
 - detector.GetPredictionLabel(bitmapImage) returns the predicted string label
