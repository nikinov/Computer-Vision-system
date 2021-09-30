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
        private delegate List<Vector2> vectorOptimizer(Vector2[] param);
        
        private static IEnumerable<Vector2[]> GetOptimizedInners(IEnumerable<Vector2[]> inners, vectorOptimizer action) => inners.Select(inner => action(inner).ToArray());
        
        private static IEnumerable<Vector2[]> GetOptimizedInners(IEnumerable<Vector2[]> inners) => inners.Select(inner => inner.ToArray());
        
        public static void Build(Vector2[] outer, List<Vector2[]> inners = null, bool outerPreprocessing = true, string innerPreprocessingType = "small")
        {
            // preprocessing
            
            if (inners != null)
            {
                if (innerPreprocessingType == "Quad")
                    inners = GetOptimizedInners(inners, ctx => SkeletonMath.GetQuadrantEdges(ctx.ToList())).ToList();
                else if (innerPreprocessingType == "Small")
                    inners = GetOptimizedInners(inners, ctx => SkeletonMath.GetEdges(ctx.ToList(), segment:2, bigLineSize:60, processSimplification: true, tolerance:5)).ToList();
                else
                    inners = GetOptimizedInners(inners).ToList();
            }

            outer = SkeletonMath.GetEdges(outer.ToList()).ToArray();
            var sk = StraightSkeleton.Generate(outer, inners);
        }
    }
}