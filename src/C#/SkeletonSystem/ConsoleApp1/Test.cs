using CGAL_StraightSkeleton_Dotnet;
using PrimitiveSvgBuilder;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Skeletonizer;

namespace ConsoleApp1
{
    class Test
    {
        public static void SkeletonTest1()
        {

            Stopwatch w = new Stopwatch();
            w.Start();

            var ssk = StraightSkeleton.Generate(new Vector2[]
            {
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
            }, new Vector2[][]
            {
                new Vector2[]
                {
                    new Vector2(2, 2),
                    new Vector2(2, -2),
                    new Vector2(-2, -2),
                    new Vector2(-2, 2)
                }
            });

            var sssk = StraightSkeleton.Generate(new Vector2[]
            {
                new Vector2(15483, -13899.6f),
                new Vector2(15481.51f, 13903.07f),
                new Vector2(13899.6f, 15483f),
                new Vector2(12917, 15483),
                new Vector2(12916.99f, -15473),
                new Vector2(12643, -15472.99f),
                new Vector2(12643, 15483),
                new Vector2(-13903.07f, 15481.5f),
                new Vector2(-15483, 13899.61f),
                new Vector2(-15481.51f, -13903.07f),
                new Vector2(-13899.6f, -15483),
                new Vector2(13903.07f, -15481.51f),
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

        public static void SkeletonTest2()
        {
            List<List<Vector2>> polygons =
                SkeletonMath.GetVerts("../../../../../../resources/CoordinateData/pythonCoordinates.txt");
            List<Vector2[]> optimizedPolygons = new List<Vector2[]>();
            //List<List<Vector2d>> optimizedPolygons2 = new List<List<Vector2d>>();

            int iter = 0;
            foreach (var pol in polygons)
            {
                if (iter == 0)
                {
                    optimizedPolygons.Add(SkeletonMath.GetEdges(pol).ToArray());
                }
                else
                {
                    if (pol.Count != 0)
                    {
                        //optimizedPolygons.Add(SkeletonTestUtil.GetEdges(pol, segment:2, bigLineSize:60, processSimplification: true, tolerance:5));
                        //optimizedPolygons.Add(SkeletonTestUtil.GetQuadrantEdges(pol));
                        var pol2 = new List<Vector2>();
                        for (int i = pol.Count - 1; i >= 0; i--)
                        {
                            pol2.Add(pol[i]);
                        }

                        optimizedPolygons.Add(SkeletonMath.GetEdges(pol, segment: 10, bigLineSize: 30,
                            processSimplification: true, tolerance: 20).ToArray());
                    }
                }

                iter += 1;
            }

            List<Vector2> full = new List<Vector2>();

            var outer = optimizedPolygons[0];
            optimizedPolygons.RemoveAt(0);
            List<Vector2> outer2 = new List<Vector2>();

            for (int i = outer.Length - 1; i >= 0; i--)
            {
                outer2.Add(outer[i]);
            }

            var sk = StraightSkeleton.Generate(outer2.ToArray(), optimizedPolygons.ToArray());
            string text = "";
            foreach (var edge in sk.Skeleton)
            {
                full = new List<Vector2>();
                full.Add(edge.Start.Position);
                full.Add(edge.End.Position);
                text = text + SkeletonMath.SaveGeometry(full.ToArray()) + "\n";
            }

            File.WriteAllText("../../../../../../resources/CoordinateData/skeletonLines.txt", text);
        }

        public static void SkeletonTest3()
        {
            List<List<Vector2>> polygons = SkeletonMath.GetVerts("../../../../../../resources/CoordinateData/pythonCoordinates.txt");
            List<Vector2> full = new List<Vector2>();
            string text = "";
            foreach (var pol in polygons)
            {
                //optimizedPolygons.Add(SkeletonTestUtil.GetEdges(pol, segment:2, bigLineSize:60, processSimplification: true, tolerance:5));
                //optimizedPolygons.Add(SkeletonTestUtil.GetQuadrantEdges(pol));
                var pol2 = new List<Vector2>();
                for (int i = pol.Count - 1; i >= 0; i--)
                {
                    pol2.Add(pol[i]);
                }

                var sk = StraightSkeleton.Generate(SkeletonMath.GetQuadrantEdges(pol2).ToArray());

                foreach (var edge in sk.Skeleton)
                {
                    full = new List<Vector2>();
                    full.Add(edge.Start.Position);
                    full.Add(edge.End.Position);
                    text = text + SkeletonMath.SaveGeometry(full.ToArray()) + "\n";
                }
            }
            File.WriteAllText("../../../../../../resources/CoordinateData/skeletonLines.txt", text);
        }
    }
}
