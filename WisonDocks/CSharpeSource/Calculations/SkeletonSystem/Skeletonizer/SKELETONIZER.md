# Skeletonizer

This project has a lot of functions for mesh optimization
as well as methods for constructing [Straight Skeleton](https://en.wikipedia.org/wiki/Straight_skeleton).
___
## Skeleton maker
### overview
To get started with building your first skeleton you can
reference the Skeletonizer library and then after you have
your polygon made you can format it in a form of a Vector2
array. 

### optional params
For optional parameters if you have wholes format them in a form of a List of Vector2
arrays where every Array is a whole. `inners: myInners`

You can also specify preprocessing for outer meaning the outer
polygon will be optimized to the smallest number of vertexes. `outerPreprocessing: false`

And finally for optimizing the inner polygons you can specify an empty string`innerPreprocessingType: ""`
if you don't want any optimizations for the inner polygons 
in case you want optimizations you can specify `innerPreprocessingType: "small"` for some
light preprocessing or `innerPreprocessingType: "quad"` if the wholes will only have 4 corners.
it's small by default.
### code

The `SkeletonMaker.Build` function will return 

```cs
// create my outer polygon
Vector2[] myOuterPolygon = new[]
{
    new Vector2(0, 0),
    new Vector2(0, 10),
    new Vector2(5, 10),
    new Vector2(5, 0)
};

// create my inner polygon
Vector2[] myInnerPolygon1 = new[]
{
    new Vector2(1, 1),
    new Vector2(1, 4),
    new Vector2(4, 4),
    new Vector2(4, 1)
};

Vector2[] myInnerPolygon2 = new[]
{
    new Vector2(1, 6),
    new Vector2(1, 9),
    new Vector2(4, 9),
    new Vector2(4, 6)
};

List<Vector2[]> myInnerPolygons = new List<Vector2[]>();
myInnerPolygons.Add(myInnerPolygon1);
myInnerPolygons.Add(myInnerPolygon2);

// build my skeleton with no preprocessing
List<Vector2[]> mySkeleton = SkeletonMaker.Build(myOuterPolygon, myInnerPolygons, outerPreprocessing: false, innerPreprocessingType: "");
```

## Skeleton math


