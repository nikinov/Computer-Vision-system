using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using NUnit.Framework.Constraints;
using StraightSkeletonNet.Primitives;

namespace StraightSkeletonNet.Tests
{
    public static class SkeletonTestUtil
    {
        public static List<Vector2d> GetFacePoints(Skeleton sk)
        {
            List<Vector2d> ret = new List<Vector2d>();

            foreach (EdgeResult edgeOutput in sk.Edges)
            {
                List<Vector2d> points = edgeOutput.Polygon;
                foreach (Vector2d vector2d in points)
                {
                    if (!ContainsEpsilon(ret, vector2d))
                        ret.Add(vector2d);
                }
            }
            return ret;
        }

        public static void AssertExpectedPoints(List<Vector2d> expectedList, List<Vector2d> givenList)
        {
            StringBuilder sb = new StringBuilder();
            foreach (Vector2d expected in expectedList)
            {
                if (!ContainsEpsilon(givenList, expected))
                    sb.AppendFormat("Can't find expected point ({0}, {1}) in given list\n", expected.X, expected.Y);
            }

            foreach (Vector2d given in givenList)
            {
                if (!ContainsEpsilon(expectedList, given))
                    sb.AppendFormat("Can't find given point ({0}, {1}) in expected list\n", given.X, given.Y);
            }

            if (sb.Length > 0)
                throw new InvalidOperationException(sb.ToString());
        }

        public static bool ContainsEpsilon(List<Vector2d> list, Vector2d p)
        {
            return list.Any(l => EqualEpsilon(l.X, p.X) && EqualEpsilon(l.Y, p.Y));
        }

        public static bool EqualEpsilon(double d1, double d2)
        {
            return Math.Abs(d1 - d2) < 5E-6;
        }

        public static List<List<Vector2d>> GetVerts(string textFile)
        {
            string text = File.ReadAllText(textFile);

            List<List<Vector2d>> Verts = new List<List<Vector2d>>();
            foreach (var polygon in text.Split('\n'))
            {
                List<Vector2d> pol = new List<Vector2d>();
                foreach (var vertex in polygon.Split(','))
                {
                    if (vertex != "")
                    {
                        var vert = vertex.Split('|');
                        pol.Add(new Vector2d(Double.Parse(vert[0]), Double.Parse(vert[1])));
                    }
                }
                Verts.Add(pol);
            }
            return Verts;
        }

        // ----------------------------
        //
        // this is where my code starts
        //
        // ----------------------------

        public static string SaveGeometry(List<Vector2d> vert)
        {
            string text = "";

            foreach (var ver in vert)
            {
                text = text + ver.X + "|" + ver.Y + ",";
            }

            return text;
        }
        // returns square of distance b/w two points
        static double lengthSquare(Vector2d p1, Vector2d p2)
        {
            var xDiff = p1.X - p2.X;
            var yDiff = p1.Y - p2.Y;
            return xDiff * xDiff + yDiff * yDiff;
        }
        public static List<List<T>> ChunkBy<T>(this List<T> source, int chunkSize)
        {
            return source
                .Select((x, i) => new { Index = i, Value = x })
                .GroupBy(x => x.Index / chunkSize)
                .Select(x => x.Select(v => v.Value).ToList())
                .ToList();
        }

        public static int Mod(int num, int mod)
        {
            return num % mod;
        }
        public static bool IsClose(List<float> angles, float tolerance=0.1f)
        {
            foreach (var ang in angles)
            {
                if (ang < angles[0] + tolerance && ang > angles[0] - tolerance) { }
                else return false;
            }
            return true;
        }

        public static List<T> Select<T>(List<T> fullList, int start, int take)
        {
            return fullList.Skip(start).Take(take).ToList();
        }

        public static List<Vector2d> FindBigLines(List<Vector2d> allPoints, float bigLineSize=10)
        {
            List<Vector2d> edges = new List<Vector2d>();

            for (int i = 0; i < allPoints.Count; i++)
            {
                int next;
                if (i == allPoints.Count - 1)
                    next = 0;
                else
                    next = i + 1;
                if (Vector2.Distance(new Vector2((float)allPoints[i].X, (float)allPoints[i].Y), new Vector2((float)allPoints[next].X, (float)allPoints[next].Y)) > 50)
                {
                    edges.Add(allPoints[i]);
                }
            }
            return edges;
        }

        public static bool CheckForSymetry(float angle, float tolerance)
        {
            List<float> syms = new List<float>()
            {
                -180,
                -135,
                -90,
                -45,
                0,
                45,
                90,
                135,
                180
            };

            foreach (var sym in syms)
            {
                if (angle < sym + tolerance && angle > sym - tolerance)
                    return true;
            }
            return false;
        }

        public static Vector2d GetMeshCenter(List<Vector2d> mesh)
        {
            Vector2d center = new Vector2d();
            double[] posittions = new double[4];
            
            foreach (var pos in mesh)
            {
                if (pos.X < posittions[0])
                    posittions[0] = pos.X;
                else if (pos.X > posittions[1])
                    posittions[1] = pos.X;
                else if (pos.Y < posittions[2])
                    posittions[2] = pos.Y;
                else if (pos.Y > posittions[3])
                    posittions[3] = pos.Y;
            }

            var yVal = posittions[1] + (posittions[0] - posittions[1]) / 2;
            var xVal = posittions[3] + (posittions[2] - posittions[3]) / 2;

            center.X = xVal;
            center.Y = yVal;

            return center;
        }

        public static float GetDirection(Vector2d startPoint, Vector2d endPoint)
        {
            double xDiff = endPoint.X - startPoint.X;
            double yDiff = endPoint.Y - startPoint.Y;
            return (float)(Math.Atan2(yDiff, xDiff) * 180.0 / Math.PI);
        }
        
        public static List<int> CheckForInconsistency(List<float>segment, int tolerance=5)
        {
            List<int> indexes = new List<int>();

            var prevSeg = segment[segment.Count - 1];
            int iter = 0;
            int leftTolarance = -180 + tolerance;
            int rightTolarance = 180 - tolerance;

            foreach (var seg in segment)
            {
                if (seg + tolerance > prevSeg && seg - tolerance < prevSeg) { }
                else
                {
                    if ((prevSeg < leftTolarance && seg > rightTolarance && 180 / seg + 180 + prevSeg < tolerance) || (prevSeg > leftTolarance && seg < rightTolarance && 180 / prevSeg + 180 + seg < tolerance)) { }
                    else
                        indexes.Add(iter);
                }
                prevSeg = seg;
                iter++;
            }
            return indexes;
        }

        public static List<float> ConvertIntoDirections(List<Vector2d> points)
        {
            List<float> directions = new List<float>();

            for (int i = 0; i < points.Count; i++)
            {
                if (i == points.Count-1)
                {
                    directions.Add(GetDirection(points[i], points[0]));
                }
                else
                {
                    directions.Add(GetDirection(points[i], points[i+1]));
                }
            }
            return directions;
        }

        public static List<float> GetSegments(List<float> segment, int chunkSize)
        {
            List<List<float>> segments = ChunkBy<float>(segment, chunkSize);
            List<float> outSegment = new List<float>();

            foreach (var seg in segments)
            {
                outSegment.Add(seg.Average());
            }
            return outSegment;
        }


        public static float GetAvaragePointDistance(Vector2d mainPoint, List<Vector2d> otherPoints)
        {
            Vector2 mainPointV = new Vector2((float)mainPoint.X, (float)mainPoint.Y);

            float[] allDistances = new float[otherPoints.Count];

            int count = 0;
            foreach (var point in otherPoints)
            {
                Vector2 pointV = new Vector2((float)point.X, (float)point.Y);
                allDistances[count] = Vector2.Distance(pointV, mainPointV);
                count++;
            }

            return allDistances.Average();
        }
        
        public static void FindIntersection(
            Vector2d p1, Vector2d p2, Vector2d p3, Vector2d p4,
            out bool lines_intersect, out bool segments_intersect,
            out Vector2d intersection,
            out Vector2d close_p1, out Vector2d close_p2)
        {
            // Get the segments' parameters.
            var dx12 = p2.X - p1.X;
            var dy12 = p2.Y - p1.Y;
            var dx34 = p4.X - p3.X;
            var dy34 = p4.Y - p3.Y;

            // Solve for t1 and t2
            var denominator = (dy12 * dx34 - dx12 * dy34);

            var t1 = ((p1.X - p3.X) * dy34 + (p3.Y - p1.Y) * dx34) / denominator;
            if (float.IsInfinity((float)t1))
            {
                // The lines are parallel (or close enough to it).
                lines_intersect = false;
                segments_intersect = false;
                intersection = new Vector2d(float.NaN, float.NaN);
                close_p1 = new Vector2d(float.NaN, float.NaN);
                close_p2 = new Vector2d(float.NaN, float.NaN);
                return;
            }
            lines_intersect = true;

            var t2 = ((p3.X - p1.X) * dy12 + (p1.Y - p3.Y) * dx12) / -denominator;

            // Find the point of intersection.
            intersection = new Vector2d(p1.X + dx12 * t1, p1.Y + dy12 * t1);

            // The segments intersect if t1 and t2 are between 0 and 1.
            segments_intersect =
                ((t1 >= 0) && (t1 <= 1) &&
                 (t2 >= 0) && (t2 <= 1));

            // Find the closest points on the segments.
            if (t1 < 0)
            {
                t1 = 0;
            }
            else if (t1 > 1)
            {
                t1 = 1;
            }

            if (t2 < 0)
            {
                t2 = 0;
            }
            else if (t2 > 1)
            {
                t2 = 1;
            }

            close_p1 = new Vector2d(p1.X + dx12 * t1, p1.Y + dy12 * t1);
            close_p2 = new Vector2d(p3.X + dx34 * t2, p3.Y + dy34 * t2);
        }

        public static float GetAngle(Vector2d A, Vector2d B, Vector2d P)
        {
            // Square of lengths be a2, b2, c2
            double a2 = lengthSquare(B, P);
            double b2 = lengthSquare(A, P);
            double P2 = lengthSquare(A, B);

            // length of sides be a, b, c
            float a = (float)Math.Sqrt(a2);
            float b = (float)Math.Sqrt(b2);

            // From Cosine law
            float gamma = (float)Math.Acos((a2 + b2 - P2) /
                                               (2 * a * b));
            if (gamma.ToString() == float.NaN.ToString())
            {
                if (a == b)
                {
                    return 180;
                }
                else
                {
                    return 0;
                }
            }

            // Converting to degree
            gamma = (float)(gamma * 180 / Math.PI);

            return gamma;
        }

        public static Vector2 GetClosestPointOnLineSegment(Vector2 a, Vector2 b, Vector2 p)
        {
            Vector2 ap = p - a;       //Vector from A to P   
            Vector2 ab = b - a;       //Vector from A to B  

            float magnitudeAb = ab.LengthSquared();     //Magnitude of AB vector (it's length squared)     
            float abaPproduct = Vector2.Dot(ap, ab);    //The DOT product of a_to_p and a_to_b     
            float distance = abaPproduct / magnitudeAb; //The normalized "distance" from a to your closest point  

            if (distance < 0)     //Check if P projection is over vectorAB     
            {
                return a;
            }
            else if (distance > 1)
            {
                return b;
            }
            else
            {
                return a + ab * distance;
            }
        }

        public static bool SanityyCheck(Vector2d a, Vector2d b, Vector2d point)
        {
            if (GetAngle(a, b, point) < 135)
            {
                return false;
            }
            return true;
        }

        public static int GetFurthestPointIndex(List<Vector2d> line, List<Vector2d> points)
        {
            int pointCount = points.Count;
            float[] distances = new float[pointCount];

            Vector2d furtherstPoint;
            List<Vector2> newLine = new List<Vector2>()
            {
                new Vector2((float)line[0].X, (float)line[0].Y),
                new Vector2((float)line[1].X, (float)line[1].Y)
            };

            for (int i=0; i<pointCount; i++)
            {
                Vector2 P = new Vector2((float)points[i].X, (float)points[i].Y);
                distances[i] = Vector2.Distance(GetClosestPointOnLineSegment(newLine[0], newLine[1], P), P);
            }
            float largestDistance = distances.Max();

            if (SanityyCheck(line[0], line[1], points[Array.IndexOf(distances, largestDistance)]))
                return -1;
            else
                return Array.IndexOf(distances, largestDistance);
        }

        public static List<Vector2d> FindEdges(List<Vector2d> mesh, int segment=15, float bigLineSize=100)
        {
            List<Vector2d> edges = new List<Vector2d>();
            var inconsistency = CheckForInconsistency(GetSegments(ConvertIntoDirections(mesh), segment), tolerance: 2);
            foreach (var it in inconsistency)
            {
                foreach (var seg in Select<Vector2d>(mesh, it*segment, segment))
                    edges.Add(seg);
            }
            var edg = FindBigLines(edges, bigLineSize:bigLineSize);

            var icon = CheckForInconsistency(ConvertIntoDirections(edg), tolerance: 2);
            List<Vector2d> edges2 = new List<Vector2d>();
            foreach (var it in icon)
            {
                edges2.Add(edg[it]);
            }
            
            List<int> correctionIndexes =new List<int>();
            var angles = ConvertIntoDirections(edges2);
            for (int i = 0; i < angles.Count; i++)
            {
                if (!CheckForSymetry(angles[i], 1))
                {
                    correctionIndexes.Add(i);
                }
            }

            var newEd = edges2;
            var offset = -1;
            for (int i = edges2.Count-1; i >= 0; i--)
            {
                if (correctionIndexes.Contains(i))
                {
                    bool lines_intersect;
                    bool segments_intersect;
                    Vector2d intersection;
                    Vector2d close_p1;
                    Vector2d close_p2;
                    FindIntersection(edges2[i + offset], edges2[Mod(i+1 + offset, edges2.Count-1)], edges2[Mod(i+3 + offset, edges2.Count-1)], edges2[Mod(i+2 + offset, edges2.Count-1)], out lines_intersect, out segments_intersect, out intersection, out close_p1, out close_p2);
                    if (lines_intersect)
                        newEd[newEd.IndexOf(edges2[Mod(i + 1 + offset, edges2.Count - 1)])] = intersection;
                    newEd.Remove(edges2[Mod(i + 2 + offset, edges2.Count - 1)]);
                }
            }
            
            File.WriteAllText("../../../../../../resources/CoordinateData/generatedTest.txt", SaveGeometry(edges2));

            return edges2;
        }
        public static List<Vector2d> optimizeMesh(int newMeshCount, List<Vector2d> mesh)
        {
            int oldMeshLength = mesh.Count;
            int meshChunkLength = oldMeshLength / newMeshCount;
            int[] indexes = new int[newMeshCount];
            List<Vector2d> newMesh = new List<Vector2d>();
            int corrects = 0;

            for (int i = 0; i < newMeshCount; i++)
            {
                indexes[i] = meshChunkLength * i;
                newMesh.Add(mesh[indexes[i]]);
            }

            var initText = SaveGeometry(newMesh);
            File.WriteAllText("../../../../../../resources/CoordinateData/initOptimizedMesh.txt", initText);

            var text = "";
            int iterr = 0;
            while (corrects != newMeshCount)
            {
                newMesh = new List<Vector2d>();
                corrects = 0;
                for (int i=0; i<newMeshCount; i++)
                {
                    List<Vector2d> selectedPoints = mesh.Skip(meshChunkLength * i).Take(meshChunkLength).ToList();
                    if (i == newMeshCount - 1)
                    {
                        var idx = GetFurthestPointIndex(new List<Vector2d>() { mesh[indexes[i]], mesh[indexes[i] * 0] }, selectedPoints);
                        if (idx != -1)
                            indexes[i] = idx;
                        else
                            corrects += 1;
                    }
                    else
                    {
                        var idx = GetFurthestPointIndex(new List<Vector2d>() { mesh[indexes[i]], mesh[indexes[i + 1]] }, selectedPoints);
                        if (idx != -1)
                            indexes[i] = idx;
                        else
                            corrects += 1;
                    }
                }
                iterr += 1;
                if (iterr > newMeshCount)
                    break;
                foreach (var idx in indexes)
                {
                    newMesh.Add(mesh[idx]);
                }
                text = text + SaveGeometry(newMesh) + "\n";
            }

            File.WriteAllText("../../../../../../resources/CoordinateData/optimizedMesh.txt", text);

            return newMesh;
        }
    }
}