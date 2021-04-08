using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClassLibraryDetector
{
    class ModelInput
    {
        [ImageType(300, 300)]
        public Bitmap image { get; set; }
    }
}
