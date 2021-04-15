The following NuGet packages must be installed to OnnxDetector project:
	"Microsoft.ML" version="1.5.5"
	"Microsoft.Rest.ClientRuntime.Azure" version="3.3.18"
	"Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction" version="2.0.0"
	"Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training" version="2.0.0"
	"Microsoft.ML.ImageAnalytics" version="1.5.5"
	"Microsoft.ML.OnnxRuntime"
	"Microsoft.ML.OnnxTransformer" version="1.5.5"
	(requires protobuf 3.10.1)
	Then update ProtoBuf to the last version!


The following NuGet packages must be installed to project which uses OnnxDetector:
	Google.Protobuf (>3.11.4)



If exception "Unable to load DLL 'onnxruntime'" occurs then copy the onnxruntime.dll manually to bin folder.
(or reinstall Microsoft.ML.OnnxRuntime NuGet package)