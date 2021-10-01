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
        
        private static IEnumerable<Vector2[]> GetOptimizedInners(IEnumerable<Vector2[]> inners, vectorOptimizer action) => inners.Where(inner => inner.Length != 0).Select(inner => action(inner).ToArray());
        
        private static IEnumerable<Vector2[]> GetOptimizedInners(IEnumerable<Vector2[]> inners) => inners.Select(inner => inner.ToArray());

        public static void Build(Vector2[] outer, List<Vector2[]> inners = null, bool outerPreprocessing = true, string innerPreprocessingType = "small")
        {
            // preprocessing

            if (inners != null)
            {
                switch (innerPreprocessingType)
                {
                    case "quad":
                        inners = GetOptimizedInners(inners, ctx => SkeletonMath.GetQuadrantEdges(ctx.ToList())).ToList();
                        break;
                    case "small":
                        inners = GetOptimizedInners(inners,
                            ctx => SkeletonMath.GetEdges(ctx.ToList(), 2, bigLineSize: 60,
                                processSimplification: true, tolerance: 5)).ToList();
                        break;
                    default:
                        inners = GetOptimizedInners(inners).ToList();
                        break;
                }
                for (var i = 0; i < inners.Count; i++)
                {
                    if (SkeletonMath.IsClockwise(inners[i]))
                        inners[i] = SkeletonMath.ReverseList(inners[i].ToList()).ToArray();
                }
            }


            if (outerPreprocessing)
                outer = SkeletonMath.GetEdges(outer.ToList()).ToArray();
            
            var sk = StraightSkeleton.Generate(outer, inners);
        }
    }
}