using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using NUnit.Framework;
using StraightSkeletonNet.Primitives;
using StraightSkeletonNet.Tests;

namespace StraightSkeletonNet.Tests
{
    public class SkeletonMaker
    {
        public static List<List<Vector2d>> GetCouples(List<List<Vector2d>> inners)
        {
            List<List<Vector2d>> couples = new List<List<Vector2d>>();

            for (int i = 0; i < inners.Count; i++)
            {
                // heigh to low
                var otherSignificant = inners[i].OrderBy(v => v.X).ToArray();
                List<Vector2d> rightSigs = new List<Vector2d>()
                {
                    otherSignificant[0],
                    otherSignificant[1]
                };
                rightSigs = rightSigs.OrderBy(v => v.Y).ToList();
                List<Vector2d> leftSigs = new List<Vector2d>()
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
            
            // try
            return couples;
        }

        public static List<List<List<Vector2d>>> optimizeCouples(List<List<Vector2d>> couples, List<Vector2d> outer)
        {
            List<List<List<Vector2d>>> optimizedCouples = new List<List<List<Vector2d>>>();
            List<List<Vector2d>> coupleTemp = new List<List<Vector2d>>();
            foreach (var couple in couples)
            {
                if (SkeletonTestUtil.PointInPolygon(SkeletonTestUtil.GetMeshCenter(couple), outer))
                {
                    optimizedCouples.Add(coupleTemp);
                    coupleTemp = new List<List<Vector2d>>();
                }
                else
                    coupleTemp.Add(couple);
            }
            
            return optimizedCouples;
        }

        public static List<Vector2d> GetFingers(List<List<List<Vector2d>>> couplesGroups, List<Vector2d> outer)
        {
            foreach (var couples in couplesGroups)
            {
                foreach (var couple in couples)
                {
                    
                }
            }
            return 
        }

        public static List<Vector2d> BuildSkeleton(List<Vector2d> outer, List<List<Vector2d>> inners)
        {
            List<Vector2d> skeleton = new List<Vector2d>();
            
            return skeleton;
        }
    }
}