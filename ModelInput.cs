using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace ONNXDetector
{
    class ModelInput
    {
        [ImageType(300, 300)]
        public Bitmap image { get; set; }
    }
}
