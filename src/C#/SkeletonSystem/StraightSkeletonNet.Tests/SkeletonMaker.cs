using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NUnit.Framework;
using StraightSkeletonNet.Primitives;
using StraightSkeletonNet.Tests;

namespace StraightSkeletonNet.Tests
{
    public class SkeletonMaker
    {
        public static float GetWidth(List<Vector2d> outer, List<List<Vector2d>> inners)
        {
            var yHeighInner = inners[0][0].Y;
            var yHeighOuter = outer[0].Y;
            foreach (var inner in inners)
            {
                foreach (var point in inner)
                {
                    if (yHeighInner < point.Y)
                        yHeighInner = point.Y;
                }
            }

            foreach (var point in outer)
            {
                if (yHeighInner < point.Y)
                    yHeighInner = point.Y;
            }

            if (yHeighInner >= yHeighOuter)
                throw new InvalidOperationException("the heighest point in the outer polygon is lower the the heighest point in the finger polygons");
            return (float)(yHeighOuter - yHeighInner);
        }

        public static bool GetInverse()
        {
            
        }

        public static Vector2d GetAnglePoint(Vector2d[] corner, float distance, bool inverse)
        {
            float radiansA = SkeletonTestUtil.GetDirection(corner[1], corner[0]);
            float radiansB = SkeletonTestUtil.GetDirection(corner[1], corner[2]);

            float radians = radiansA + radiansB;
            
            Vector2 vectorDirection = new Vector2((float)Math.Cos(radians), (float)Math.Sin(radians));
            
            return new Vector2d();
        }

        public static List<Vector2d> GetOuterSkeleton(List<Vector2d> outer, float length)
        {
            List<float> directionsLeft = new List<float>();
            List<float> directionsRight = new List<float>();
            for (int i = 0; i < outer.Count; i++)
            {
                directionsLeft.Add(SkeletonTestUtil.GetDirection(outer[i], outer[SkeletonTestUtil.Mod(i+1, outer.Count)]));
                directionsRight.Add(SkeletonTestUtil.GetDirection(outer[i], outer[SkeletonTestUtil.Mod(i-1, outer.Count)]));
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