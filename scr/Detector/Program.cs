using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ModelMaker;

namespace Detector
{
    class Program
    {
        static void Main(string[] args)
        {
            Maker.MakeModel();
            Console.ReadKey();
        }
    }
}
