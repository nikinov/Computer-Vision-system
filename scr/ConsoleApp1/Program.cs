using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            execProcess();
            Console.ReadKey();
        }

        static void execProcess(string pythonPath= "C:/Users/Ryzen7-EXT/anaconda3/envs/tensor/pythonw.exe", string scriptPath= "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/python/cli_example.py")
        {
            // create process info
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = pythonPath;

            var dataPath = @"C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/Assets";
            var outPath = @"C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech";

            psi.Arguments = scriptPath + " " + dataPath + " " + outPath;


            // process configuration
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;

            string error;
            string result;

            using(var process = Process.Start(psi))
            {
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
