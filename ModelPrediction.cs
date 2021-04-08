using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClassLibraryDetector
{
    class ModelPrediction
    {
        [ColumnName("model_output")]
        public float[] PredictedLabels { get; set; }
    }
}
