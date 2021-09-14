using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StraightSkeletonNet.Tests;
using StraightSkeletonNet;
using StraightSkeletonNet.Primitives;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            SkeletonTest skeletonTest = new SkeletonTest();
            skeletonTest.SkeletonTest_hole_4();
            /*
            List<Vector2d> line = new List<Vector2d>()
            {
                new Vector2d(0, 0),
                new Vector2d(100, 100),
            };
            List<Vector2d> points = new List<Vector2d>()
            {
                new Vector2d(2.3, 2),
                new Vector2d(70, 69),
                new Vector2d(59, 57),
                new Vector2d(70.2, 69.5),
                new Vector2d(59, 60),
                new Vector2d(49, 50),
                new Vector2d(49.3, 49),
            };

            Console.WriteLine(points[SkeletonTestUtil.GetFurthestPointIndex(line, points)]);
*/
            Console.ReadKey();
        }
    }
}
