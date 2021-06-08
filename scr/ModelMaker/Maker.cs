using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;
using Wision.CppRoutinesWrapper;

namespace ModelMaker
{
    public class Maker
    {
        /// <summary>
        /// Function interacts with a python training script and makes a model.pt file
        /// </summary>
        /// <param name="pythonPath">python path</param>
        /// <param name="scriptPath">the pathe to the cli python intervafe script</param>
        /// <param name="dataPath">path to your data or assets</param>
        /// <param name="outPath">the path where the model.pt file will be outputed</param>
        /// <param name="epoch">how many times will the training script go throught the data</param>
        /// <param name="save_config">if we want to save data config into a txt file</param>
        /// <param name="use_config">if we want to use a txt file with a data config</param>
        public static void MakeModel(string pythonPath = "C:/Users/Ryzen7-EXT/anaconda3/envs/tensor/pythonw.exe",
            string scriptPath = "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/python/cli_example.py",
            string dataPath = "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/Assets",
            string outPath = "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech",
            int epoch = 0,
            bool save_config=false,
            bool use_config=false)
        {
            if (dataPath == outPath)
            {
                throw new ArgumentException("you cannot have the same dataPath and outPath, specify different paths ");
            }
            // create process info
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = pythonPath;

            Console.WriteLine("arguments in");
            psi.Arguments = scriptPath + " " + dataPath + " " + outPath + " " + epoch + " " + save_config + " " + use_config;

            // process configuration
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;

            string error;
            string result;
            Console.WriteLine("config done");

            using (var process = Process.Start(psi))
            {
                Console.WriteLine("Started Training");
                error = process.StandardError.ReadToEnd();
                result = process.StandardOutput.ReadToEnd();
            }
            Console.WriteLine("Errors:");
            Console.WriteLine(error);
            Console.WriteLine("Res:");
            Console.WriteLine(result);
        }
    }
    public class Predictor
    {

        [DllImport(@"C:\Users\Ryzen7-EXT\Documents\Github\WickonHightech\resources\C++\build\PredictorDll\Debug\Predictor_dll.dll")]
        private static extern int DLL_GetPrediction(
            byte[] modelPath,
            IntPtr byteArray,
            IntPtr buffer,
            int allocSizOfBuffer
            );

        public static float[] GetPrediction(
            string modelPath,
            byte[] byteArray,
            int sizeOutput
            )
        {
            var bufferFolder = Encoding.ASCII.GetBytes(modelPath);
            var output = new float[sizeOutput];

            using (var pinbyteArray = byteArray.Pin())
            using (var pinOutput = output.Pin())
            {
                DLL_GetPrediction(bufferFolder, pinbyteArray, pinOutput, sizeOutput);
            }
            return output;
        }

    }
}
