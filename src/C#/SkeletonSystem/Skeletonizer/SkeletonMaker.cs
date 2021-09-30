using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using CGAL_StraightSkeleton_Dotnet;
using NUnit.Framework;

namespace Skeletonizer
{
    public class SkeletonMaker
    {
        public static void Build(List<Vector2> outer, List<List<Vector2>> inners = null, bool outerPreprocessing = true, bool innerPreprocessing = true)
        {
            // preprocessing
            List<Vector2[]> optimizedInners;
            if (inners != null)
            {
                optimizedInners = new List<Vector2[]>();
                foreach (var inner in inners)
                {
                    optimizedInners.Add(SkeletonMath.GetEdges(inner).ToArray());
                }
            }

            var optimizedOuter = SkeletonMath.GetEdges(outer);
            //var sk = StraightSkeleton.Generate()
        }
    }
}