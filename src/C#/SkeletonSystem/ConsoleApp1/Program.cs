using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using CGAL_StraightSkeleton_Dotnet;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
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
            // stuff here is for CGAL

            Stopwatch w = new Stopwatch();
            w.Start();

            var ssk = StraightSkeleton.Generate(new Vector2[] {
                new Vector2(-10, 10),
                new Vector2(-1, 10),
                new Vector2(0, 11),
                new Vector2(1, 10),
                new Vector2(10, 10),
                new Vector2(10, -2),
                new Vector2(5, 0),
                new Vector2(7, -10),
                new Vector2(-7, -10),
                new Vector2(-5, 0),
                new Vector2(-10, -2),
            }, new Vector2[][] {
                new Vector2[] {
                    new Vector2(2, 2),
                    new Vector2(2, -2),
                    new Vector2(-2, -2),
                    new Vector2(-2, 2)
                }
            });

            w.Stop();

            Stopwatch w2 = new Stopwatch();
            w2.Start();

            //Extract data from result
            var builder = new SvgBuilder(30);

            //Draw outline
            foreach (var edge in ssk.Borders)
            {
                builder.Circle(edge.Start.Position, 0.2f, "blue");
                builder.Circle(edge.End.Position, 0.2f, "blue");
                builder.Line(edge.Start.Position, edge.End.Position, 2, "blue");
            }

            //Draw offsets
            for (var i = 1; i < 10; i++)
            {
                var offset = ssk.Offset(i / 2f);
                foreach (var polygon in offset)
                    builder.Outline(polygon, "green");
            }

            //Draw spokes
            foreach (var edge in ssk.Spokes)
                builder.Line(edge.Start.Position, edge.End.Position, 4, "lime");

            //Draw straight skeleton
            foreach (var edge in ssk.Skeleton)
            {
                builder.Circle(edge.Start.Position, 0.2f, "hotpink");
                builder.Circle(edge.End.Position, 0.2f, "hotpink");
                builder.Line(edge.Start.Position, edge.End.Position, 2, "hotpink");
            }

            Console.WriteLine(builder);
            Console.Title = string.Format("Elapsed: {0}ms {1}ms", w.ElapsedMilliseconds, w2.ElapsedMilliseconds);
            Console.ReadLine();



            Console.ReadKey();
        }
    }
}
