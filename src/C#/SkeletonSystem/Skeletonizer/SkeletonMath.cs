
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using NUnit.Framework.Constraints;


namespace Skeletonizer
{
    public class SkeletonMath
    {
        public static List<Vector2[]> GetVerts(string textFile)
        {
            var text = File.ReadAllText(textFile);

            return text.Split('\n').Select(polygon => (from vertex in polygon.Split(',') where vertex != "" && vertex != "\r" select vertex.Split('|') into vertex select new Vector2(float.Parse(vertex[0]), float.Parse(vertex[1]))).ToArray()).ToList();
        }
        public static string SaveGeometry(Vector2[] vert)
        {
            return vert.Aggregate("", (current, ver) => current + ver.X + "|" + ver.Y + ",");
        }

        // returns square of distance b/w two points
        static float LengthSquare(Vector2 p1, Vector2 p2)
        {
            var xDiff = p1.X - p2.X;
            var yDiff = p1.Y - p2.Y;
            return xDiff * xDiff + yDiff * yDiff;
        }
        public static List<List<T>> ChunkBy<T>(IEnumerable<T> source, int chunkSize)
        {
            return source
                .Select((x, i) => new { Index = i, Value = x })
                .GroupBy(x => x.Index / chunkSize)
                .Select(x => x.Select(v => v.Value).ToList())
                .ToList();
        }

        public static int Mod(int x, int m)
        {
            return (x % m + m) % m;
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

        public static List<T> Select<T>(IEnumerable<T> fullList, int start, int take)
        {
            return fullList.Skip(start).Take(take).ToList();
        }

        public static List<Vector2> GetBigLines(List<Vector2> allPoints, float bigLine=50)
        {
            var edges = new List<Vector2>();

            for (var i = 0; i < allPoints.Count; i++)
            {
                int next;
                if (i == allPoints.Count - 1)
                    next = 0;
                else
                    next = i + 1;
                if (Vector2.Distance(new Vector2(allPoints[i].X, allPoints[i].Y), new Vector2(allPoints[next].X, allPoints[next].Y)) > bigLine)
                {
                    edges.Add(allPoints[i]);
                }
            }
            return edges;
        }

        public static bool GetSymetryCheck(float angle, float tolerance)
        {
            var syms = new List<float>
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

            return syms.Any(sym => angle < sym + tolerance && angle > sym - tolerance);
        }

        public static Vector2 GetMeshCenter(List<Vector2> mesh)
        {
            var center = new Vector2();
            float[] positions = {
                mesh[0].X,
                mesh[0].X,
                mesh[0].Y,
                mesh[0].Y
            };
            
            foreach (var pos in mesh)
            {
                if (pos.X < positions[0])
                    positions[0] = pos.X;
                if (pos.X > positions[1])
                    positions[1] = pos.X;
                if (pos.Y < positions[2])
                    positions[2] = pos.Y;
                if (pos.Y > positions[3])
                    positions[3] = pos.Y;
            }

            var yVal = positions[0] + (positions[1] - positions[0]) / 2;
            var xVal = positions[2] + (positions[3] - positions[2]) / 2;

            center.X = yVal;
            center.Y = xVal;

            return center;
        }

        public static float GetDirection(Vector2 startPoint, Vector2 endPoint)
        {
            var xDiff = endPoint.X - startPoint.X;
            var yDiff = endPoint.Y - startPoint.Y;
            return (float)(Math.Atan2(yDiff, xDiff) * 180.0 / Math.PI);
        }
        
        public static List<int> CheckForInconsistency(List<float>segment, int tolerance=5)
        {
            var indexes = new List<int>();

            var prevSeg = segment[segment.Count - 1];
            var iter = 0;
            var leftTolerance = -180 + tolerance;
            var rightTolerance = 180 - tolerance;

            foreach (var seg in segment)
            {
                if (seg + tolerance > prevSeg && seg - tolerance < prevSeg) { }
                else
                {
                    if ((prevSeg < leftTolerance && seg > rightTolerance && 180 / seg + 180 + prevSeg < tolerance) || (prevSeg > leftTolerance && seg < rightTolerance && 180 / prevSeg + 180 + seg < tolerance)) { }
                    else
                        indexes.Add(iter);
                }
                prevSeg = seg;
                iter++;
            }
            return indexes;
        }

        public static List<float> ConvertIntoDirections(List<Vector2> points)
        {
            return points.Select((t, i) => i == points.Count - 1
                    ? GetDirection(t, points[0])
                    : GetDirection(t, points[i + 1]))
                .ToList();
        }

        public static List<float> GetSegments(IEnumerable<float> segment, int chunkSize)
        {
            var segments = ChunkBy(segment, chunkSize);

            return segments.Select(seg => seg.Average()).ToList();
        }


        public static float GetAvaragePointDistance(Vector2 mainPoint, List<Vector2> otherPoints)
        {
            var mainPointV = new Vector2(mainPoint.X, mainPoint.Y);

            var allDistances = new float[otherPoints.Count];

            var count = 0;
            foreach (var pointV in otherPoints.Select(point => new Vector2(point.X, point.Y)))
            {
                allDistances[count] = Vector2.Distance(pointV, mainPointV);
                count++;
            }

            return allDistances.Average();
        }
        
        public static void GetIntersection(
            Vector2 p1, Vector2 p2, Vector2 p3, Vector2 p4,
            out bool linesIntersect, out bool segmentsIntersect,
            out Vector2 intersection,
            out Vector2 closeP1, out Vector2 closeP2)
        {
            // Get the segments' parameters.
            var dx12 = p2.X - p1.X;
            var dy12 = p2.Y - p1.Y;
            var dx34 = p4.X - p3.X;
            var dy34 = p4.Y - p3.Y;

            // Solve for t1 and t2
            var denominator = dy12 * dx34 - dx12 * dy34;

            var t1 = ((p1.X - p3.X) * dy34 + (p3.Y - p1.Y) * dx34) / denominator;
            if (float.IsInfinity(t1))
            {
                // The lines are parallel (or close enough to it).
                linesIntersect = false;
                segmentsIntersect = false;
                intersection = new Vector2(float.NaN, float.NaN);
                closeP1 = new Vector2(float.NaN, float.NaN);
                closeP2 = new Vector2(float.NaN, float.NaN);
                return;
            }
            linesIntersect = true;

            var t2 = ((p3.X - p1.X) * dy12 + (p1.Y - p3.Y) * dx12) / -denominator;

            // Find the point of intersection.
            intersection = new Vector2(p1.X + dx12 * t1, p1.Y + dy12 * t1);

            // The segments intersect if t1 and t2 are between 0 and 1.
            segmentsIntersect =
                t1 >= 0 && t1 <= 1 &&
                t2 >= 0 && t2 <= 1;

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

            closeP1 = new Vector2(p1.X + dx12 * t1, p1.Y + dy12 * t1);
            closeP2 = new Vector2(p3.X + dx34 * t2, p3.Y + dy34 * t2);
        }

        public static float GetAngle(Vector2 a, Vector2 b, Vector2 p)
        {
            // Square of lengths be a2, b2, c2
            var a2 = LengthSquare(b, p);
            var b2 = LengthSquare(a, p);
            var p2 = LengthSquare(a, b);

            // length of sides be a, b, c
            var aa = (float)Math.Sqrt(a2);
            var bb = (float)Math.Sqrt(b2);

            // From Cosine law
            var gamma = (float)Math.Acos((a2 + b2 - p2) /
                                         (2 * aa * bb));
            if (gamma.ToString(CultureInfo.InvariantCulture) == float.NaN.ToString(CultureInfo.InvariantCulture))
            {
                return Math.Abs(aa - bb) < .1 ? 180 : 0;
            }

            // Converting to degree
            gamma = (float)(gamma * 180 / Math.PI);

            return gamma;
        }

        private static float[] Product(Vector2 a, Vector2 b, Vector2 c)
        {
            // Get the vectors' coordinates.
            var bax = a.X - b.X;
            var bay = a.Y - b.Y;
            var bcx = c.X - b.X;
            var bcy = c.Y - b.Y;

            return new[] {bax, bay, bcx, bcy};
        }
        private static float DotProduct(Vector2 a, Vector2 b, Vector2 c)
        {
            // Calculate the dot product.
            var p = Product(a, b, c);
            return p[0] * p[2] + p[1] * p[3];
        }

        public static double CrossProductLength(Vector2 a, Vector2 b, Vector2 c)
        {
            // Calculate the Z coordinate of the cross product.
            var p = Product(a, b, c);
            return p[0] * p[3] - p[1] * p[2];
        }
        public static float GetPointAngle(Vector2 a, Vector2 b, Vector2 c)
        {
            // Get the dot product.
            var dotProduct = DotProduct(a, b, c);

            // Get the cross product.
            var crossProduct = CrossProductLength(a, b, c);

            // Calculate the angle.
            return (float)Math.Atan2(crossProduct, dotProduct);
        }
        
        // Return True if the point is in the polygon.
        public static bool PointInPolygon(Vector2 point, List<Vector2> polygon)
        {
            int maxPoint = polygon.Count - 1;
            float totalAngle = GetPointAngle(polygon[maxPoint],point,polygon[0]);

            for (int i = 0; i < maxPoint; i++)
            {
                totalAngle += GetPointAngle(polygon[i],point,polygon[i + 1]);
            }

            return (Math.Abs(totalAngle) > 1);
        }

        public static Vector2 GetClosestPointOnLineSegment(Vector2 a, Vector2 b, Vector2 p)
        {
            var ap = p - a;       //Vector from A to P   
            var ab = b - a;       //Vector from A to B  

            var magnitudeAb = ab.LengthSquared();     //Magnitude of AB vector (it's length squared)     
            var abaPproduct = Vector2.Dot(ap, ab);    //The DOT product of a_to_p and a_to_b     
            var distance = abaPproduct / magnitudeAb; //The normalized "distance" from a to your closest point  

            if (distance < 0)     //Check if P projection is over vectorAB     
            {
                return a;
            }

            if (distance > 1)
            {
                return b;
            }

            return a + ab * distance;
        }
        
        public static bool GetIntersectionStatus(Vector2 a1, Vector2 a2, Vector2 b1, Vector2 b2)
        {
            int Orientation(Vector2 p, Vector2 q, Vector2 r)
            {
                // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
                // for details of below formula.
                var val = (q.Y - p.Y) * (r.X - q.X) -
                          (q.X - p.X) * (r.Y - q.Y);
 
                if (val == 0) return 0; // collinear
 
                return val > 0? 1: 2; // clock or counterclock wise
            }
            
            bool OnSegment(Vector2 p, Vector2 q, Vector2 r)
            {
                return q.X <= Math.Max(p.X, r.X) && q.X >= Math.Min(p.X, r.X) &&
                       q.Y <= Math.Max(p.Y, r.Y) && q.Y >= Math.Min(p.Y, r.Y);
            }
            
            // Find the four orientations needed for general and
            // special cases
            var o1 = Orientation(a1, a2, b1);
            var o2 = Orientation(a1, a2, b2);
            var o3 = Orientation(b1, b2, a1);
            var o4 = Orientation(b1, b2, a2);
 
            // General case
            if (o1 != o2 && o3 != o4)
                return true;
 
            // Special Cases
            // p1, q1 and p2 are collinear and p2 lies on segment p1q1
            if (o1 == 0 && OnSegment(a1, b1, a2)) return true;
 
            // p1, q1 and q2 are collinear and q2 lies on segment p1q1
            if (o2 == 0 && OnSegment(a1, b2, a2)) return true;
 
            // p2, q2 and p1 are collinear and p1 lies on segment p2q2
            if (o3 == 0 && OnSegment(b1, a1, b2)) return true;
 
            // p2, q2 and q1 are collinear and q1 lies on segment p2q2
            return o4 == 0 && OnSegment(b1, a2, b2);
        }

        public static List<List<Vector2>> GetOrderedInners(List<List<Vector2>> inners)
        {
            var innerCenters = inners.Select(GetMeshCenter).ToList();
            var side = new Vector2();
            float distance = 0;
            foreach (var innerCenter in innerCenters)
            {
                var ds = Vector2.Distance(innerCenter, innerCenters[0]);
                if (!(ds > distance)) continue;
                distance = ds;
                side = innerCenter;
            }
            inners.Sort((a, b) => Vector2.Distance(GetMeshCenter(a),
                side).CompareTo(Vector2.Distance(GetMeshCenter(b), side)));
            return inners;
        }

        // broken 
        public static List<List<Vector2>> GetInnterOutline(IEnumerable<List<Vector2>> inners, List<Vector2> outer)
        {
            var innerOutlines = new List<List<Vector2>>();
            var innersMesh = inners.SelectMany(inner => inner).ToList();

            var innerOutline = new List<Vector2> {innersMesh[0]};
            innersMesh.RemoveAt(0);
            while (innersMesh.Count >= 0)
            {
                var next = Vector2.Zero;
                var distance = Vector2.Distance(innersMesh[0], innerOutline[innerOutline.Count - 1]);
                foreach (var point in innersMesh)
                {
                    if (!(distance > Vector2.Distance(point, innerOutline[innerOutline.Count - 1]))) continue;
                    var intersectionStatuses = new List<bool>();
                    for (var i = 0; i < outer.Count-1; i++)
                    {
                        intersectionStatuses.Add(GetIntersectionStatus(outer[i], outer[Mod(i + 1, outer.Count - 1)], point,
                            innerOutline[innerOutline.Count - 1]));
                    }

                    if (intersectionStatuses.Any(c => c)) continue;
                    next = point;
                    distance = Vector2.Distance(point,
                        innerOutline[innerOutline.Count - 1]);
                }

                if (next == Vector2.Zero)
                {
                    if (innersMesh.Count == 0)
                    {
                        innerOutlines.Add(innerOutline);
                                                break;
                    }
                    
                    innerOutlines.Add(innerOutline);
                    innerOutline = new List<Vector2> {innersMesh[0]};
                }
                else
                {
                    innersMesh.Remove(next);
                    innerOutline.Add(next);
                }
            }

            return innerOutlines;
        }
        
        public static bool GetSanityCheck(Vector2 a, Vector2 b, Vector2 point)
        {
            return !(GetAngle(a, b, point) < 135);
        }

        public static int GetFurthestPointIndex(Vector2 main, List<Vector2> points)
        {
            var mainIndex = -1;
            float prevDistance = 0;
            for (var i = 0; i < points.Count; i++)
            {
                var pt = points[i];
                var distance = Vector2.Distance(main, pt);
                if (!(distance > prevDistance)) continue;
                prevDistance = distance;
                mainIndex = i;
            }
            return mainIndex;
        }

        public static int GetFurthestPointFromLineIndex(List<Vector2> line, List<Vector2> points)
        {
            var pointCount = points.Count;
            var distances = new float[pointCount];
            
            var newLine = new List<Vector2>
            {
                new Vector2(line[0].X, line[0].Y),
                new Vector2(line[1].X, line[1].Y)
            };

            for (var i=0; i<pointCount; i++)
            {
                var p = new Vector2(points[i].X, points[i].Y);
                distances[i] = Vector2.Distance(GetClosestPointOnLineSegment(newLine[0], newLine[1], p), p);
            }
            var largestDistance = distances.Max();

            if (GetSanityCheck(line[0], line[1], points[Array.IndexOf(distances, largestDistance)]))
                return -1;
            return Array.IndexOf(distances, largestDistance);
        }

        public static List<Vector2> GetQuadrantEdges(List<Vector2> mesh)
        {
            var center = GetMeshCenter(mesh);
            var quadrants = new List<List<Vector2>>();
            for (var i = 0; i < 4; i++)
            {
                quadrants.Add(new List<Vector2>());
            }
            foreach (var point in mesh)
            {
                if (point.X > center.X)
                {
                    if (point.Y > center.Y)
                        quadrants[0].Add(point);
                    else
                        quadrants[1].Add(point);
                }
                else
                {
                    if (point.Y > center.Y)
                        quadrants[2].Add(point);
                    else
                        quadrants[3].Add(point);
                }
            }

            var edges = quadrants.Select(quadrant => quadrant[GetFurthestPointIndex(center, quadrant)]).ToList();

            var temp = edges[2];
            edges[2] = edges[3];
            edges[3] = temp;
            return edges;
        }

        public static List<Vector2> GetEdges(List<Vector2> mesh, int segment = 15, bool processSimplification = false, int tolerance = 2, float bigLineSize = 50)
        {
            var inconsistency = CheckForInconsistency(GetSegments(ConvertIntoDirections(mesh), segment), tolerance);
            var edges = inconsistency.SelectMany(it => Select(mesh, it * segment, segment)).ToList();
            var edg = GetBigLines(edges, bigLine: bigLineSize);
            var edges2 = new List<Vector2>();
            if (!processSimplification)
            {
                var icon = CheckForInconsistency(ConvertIntoDirections(edg), 2);
                edges2.AddRange(icon.Select(it => edg[it]));
            }
            else
            {
                edges2 = edg;
            }

            var correctionIndexes = new List<int>();
            var angles = ConvertIntoDirections(edges2);
            for (var i = 0; i < angles.Count; i++)
            {
                if (!GetSymetryCheck(angles[i], 1))
                {
                    correctionIndexes.Add(i);
                }
            }

            var newEd = edges2;
            const int offset = -1;
            for (var i = edges2.Count - 1; i >= 0; i--)
            {
                if (!correctionIndexes.Contains(i)) continue;
                GetIntersection(edges2[Mod(i + offset, edges2.Count - 1)], edges2[Mod(i + 1 + offset, edges2.Count - 1)], edges2[Mod(i + 3 + offset, edges2.Count - 1)], edges2[Mod(i + 2 + offset, edges2.Count - 1)], out var linesIntersect, out _, out var intersection, out _, out _);
                if (linesIntersect)
                    newEd[newEd.IndexOf(edges2[Mod(i + 1 + offset, edges2.Count - 1)])] = intersection;
                newEd.Remove(edges2[Mod(i + 2 + offset, edges2.Count - 1)]);
            }

            return edges2;
        }


        public static Vector2 GetCentroid(List<Vector2> poly)
        {
            var accumulatedArea = 0.0f;
            var centerX = 0.0f;
            var centerY = 0.0f;

            for (int i = 0, j = poly.Count - 1; i < poly.Count; j = i++)
            {
                var temp = poly[i].X * poly[j].Y - poly[j].X * poly[i].Y;
                accumulatedArea += temp;
                centerX += (poly[i].X + poly[j].X) * temp;
                centerY += (poly[i].Y + poly[j].Y) * temp;
            }

            if (Math.Abs(accumulatedArea) < 1E-7f)
                return Vector2.Zero;  // Avoid division by zero

            accumulatedArea *= 3f;
            return new Vector2(centerX / accumulatedArea, centerY / accumulatedArea);
        }
        public static IEnumerable<List<Vector2>> GetCouples(IEnumerable<List<Vector2>> inners)
        {
            var couples = new List<List<Vector2>>();

            foreach (var t in inners)
            {
                // heigh to low
                var otherSignificant = t.OrderBy(v => v.X).ToArray();
                var rightSigs = new List<Vector2>()
                {
                    otherSignificant[0],
                    otherSignificant[1]
                };
                rightSigs = rightSigs.OrderBy(v => v.Y).ToList();
                var leftSigs = new List<Vector2>
                {
                    otherSignificant[2],
                    otherSignificant[3]
                };
                leftSigs = leftSigs.OrderBy(v => v.Y).ToList();
                if (couples.Count != 0)
                {
                    couples[couples.Count-1].Add(leftSigs[0]);
                    couples[couples.Count-1].Add(leftSigs[1]);
                }
                couples.Add(rightSigs);
            }
            couples.RemoveAt(couples.Count - 1);
            // try
            return couples;
        }

        public static IEnumerable<List<List<Vector2>>> OptimizeCouples(IEnumerable<List<Vector2>> couples, List<Vector2> outer)
        {
            var optimizedCouples = new List<List<List<Vector2>>>();
            var coupleTemp = new List<List<Vector2>>();
            foreach (var couple in couples)
            {
                if (PointInPolygon(GetMeshCenter(couple), outer))
                {
                    coupleTemp.Add(couple);
                }
                else
                {
                    optimizedCouples.Add(coupleTemp);
                    coupleTemp = new List<List<Vector2>>();
                }
            }
            optimizedCouples.Add(coupleTemp);
            
            return optimizedCouples;
        }

        public static List<Vector2> GetFinger(List<Vector2> couple, List<Vector2> outer)
        {
            var finger = new List<Vector2>();

            var intersectionPoints = new List<Vector2>();
            
            for (var j = 0; j < couple.Count; j++)
            {
                int other;
                switch (j)
                {
                    case 0:
                        other = 1;
                        break;
                    case 1:
                        other = 0;
                        break;
                    case 2:
                        other = 3;
                        break;
                    default:
                        other = 2;
                        break;
                }
                
                var closestPos = new Vector2();
                var closestDistance = float.PositiveInfinity;
                for (var i = 0; i < outer.Count; i++)
                {
                    
                        GetIntersection(couple[j], couple[other], outer[i], outer[Mod(i+1, outer.Count)],
                        out var linesIntersect,
                        out _,
                        out var intersection,
                        out _,
                        out _
                    );
                    if (!linesIntersect) continue;
                    var distance = Vector2.Distance(intersection,
                        couple[other]);

                    if (!(distance < closestDistance)) continue;
                    closestPos = intersection;
                    closestDistance = distance;
                }
                intersectionPoints.Add(closestPos);
            }

            var nailMesh = new List<Vector2> {intersectionPoints[0], couple[1], intersectionPoints[2], couple[3]};
            finger.Add(GetMeshCenter(nailMesh));
            var nailMesh2 = new List<Vector2> {intersectionPoints[1], couple[0], intersectionPoints[3], couple[2]};
            finger.Add(GetMeshCenter(nailMesh2));
            
            return finger;
        }

        public static List<List<Vector2>> GetFingers(List<List<Vector2>> inners, List<Vector2> outer)
        {
            var fingers = new List<List<Vector2>>();
            var couplesGroups = OptimizeCouples(GetCouples(inners), outer);
            foreach (var couples in couplesGroups)
            {
                fingers.AddRange(couples.Select(couple => GetFinger(couple, outer)));
            }

            return fingers;
        }
    }
}