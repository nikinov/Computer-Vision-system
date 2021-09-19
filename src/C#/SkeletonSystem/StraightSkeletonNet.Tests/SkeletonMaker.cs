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
            couples.RemoveAt(couples.Count - 1);
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

        public static List<Vector2d> GetFingers(List<List<Vector2d>> inners, List<Vector2d> outer)
        {
            var couplesGroups = optimizeCouples(GetCouples(inners), outer);
            foreach (var couples in couplesGroups)
            {
                foreach (var couple in couples)
                {
                    // define intersection points
                    List<Vector2d> oposites;
                    for (int j = 0; j < couple.Count; j++)
                    {
                        int other;
                        if (j == 0)
                            other = 1;
                        else if (j == 1)
                            other = 0;
                        else if (j == 2)
                            other = 3;
                        else
                            other = 2;
                        
                        var intersections = new List<Vector2d>();
                        Vector2d ClosestIter;
                        float closestDistance = float.PositiveInfinity;
                        for (int i = 0; i < outer.Count; i++)
                        {
                            bool lines_intersect;
                            bool segment_intersect;
                            Vector2d intersection;
                            Vector2d close_P1;
                            Vector2d close_P2;
                            SkeletonTestUtil.GetIntersection(couple[j], couple[other], outer[i], outer[SkeletonTestUtil.Mod(i+1, outer.Count)],
                                out lines_intersect,
                                out segment_intersect,
                                out intersection,
                                out close_P1,
                                out close_P2
                            );
                            if (lines_intersect)
                            {
                                var distance = Vector2.Distance(SkeletonTestUtil.VdToV(intersection),
                                    SkeletonTestUtil.VdToV(couple[1]));

                                if (distance < closestDistance)
                                {
                                    ClosestIter = intersection;
                                    closestDistance = distance;
                                }
                            }
                        }
                        
                    }
                }
            }

            return new List<Vector2d>();
        }

        public static List<Vector2d> BuildSkeleton(List<Vector2d> outer, List<List<Vector2d>> inners)
        {
            List<Vector2d> skeleton = new List<Vector2d>();
            
            return skeleton;
        }
    }
}