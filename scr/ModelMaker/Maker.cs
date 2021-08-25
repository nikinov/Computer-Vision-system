using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;
using RoutinesWrapper;

namespace ModelMaker
{
    public class Maker
    {
        public static void MakeModel(string pythonPath = "C:/Users/Ryzen7-EXT/anaconda3/envs/tensor/pythonw.exe",
            string scriptPath = "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/python/new_vision_system/cli_base.py",
            string dataPath = "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/Assets",
            string outPath = "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech",
            string modelName = "model",
            bool create_csv_file = false,
            bool enabled_training = true)
        {
            if (dataPath == outPath)
            {
                throw new ArgumentException("you cannot have the same dataPath and outPath, specify different paths ");
            }
            // create process info
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = pythonPath;

            Console.WriteLine("arguments in");
            psi.Arguments = scriptPath + " " + dataPath + " " + outPath + " " + modelName + " " + create_csv_file + " " + enabled_training;

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
}
