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

The `SkeletonMaker.Build` function will return a list of Vector2 arrays where every array is a
line in the skeleton with Starting and ending points.

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
This following module is made up of useful geometry and optimization functions. It was used in
the SkeletonMaker Class for mesh optimization.
___

### Saving and loading Geometry Data
You can save geometry data or a list of Vector2 arrays into a txt file and then load them.

```cs
// Define file name 
var name = "C:/myFavoriteSkeleton.txt";

// Save data
SkeletonMath.SaveGeometry(mySkeleton, name);

// Load data
List<Vector2[]> mySkeleton = SkeletonMath.GetVertsFromFile(name)
```

### Geometry functions
Simple mod
```cs
int value = SkeletonMath.Mod(x, m);
```

To slice up a List you can call the following
```cs
List<List<MyType>> Chunks = SkeletonMath.ChunkBy(ListOfMyType, chunkSize);
```

For selecting a chunk of a list
```cs
List<MyType> selecetedList = SkeletonMath.Select(ListOfMyType, startingIndex, endingIndex);
```

Getting all the lines over a certain size in a polygon, 
returns a new polygon with only the points that are connected to big lines
```cs
int = bigLineLimit = 50;
List<Vector2> BigLines = SkeletonMath.GetBigLines(myPolygon, bigLineLimit);
```

Mesh center (!not centroid!)
```cs
Vector2 center = SkeletonMath.GetMeshCenter(myPolygon);
```

Get the euler direction of a Line
```cs
flaot eulerDirection = SkeletonMath.GetDirection((Vector2)startPoint, (Vector2)endPoint);
```

Get inconsistency in a list of euler directions, will
return indexes of eulerAngles where the polygon breaks consistency
with a degree of freedom
```cs
int tolerance = 5f; // degree of freedom
List<int> indexesOfCorners = SkeletonMath.CheckForInconsisteny((List<float>)myPolygonEulerList, (flaot)tolerance);
```

Convert a polygon into a list of euler directions
```cs
List<float> myDirections = SkeletonMath.ConvertIntoDirections(myPolygon);
```

Get the intersection or predicted intersection of 2 lines
refer to further information about this algorithm [here](https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/)
```cs
SkeletonMath.GetIntersection(
    (Vector2) startOfFirstLine,
    (Vector2) endOfFirstLine,
    (Vector2) startOfSecondLine,
    (Vector2) endOfSecondLine
    out (bool) areLinesIntersecting,
    out (bool) areSegmentsIntersecting,
    out (Vector2) intersectionPoint,
    out (Vector2) firstClosestIntersectionPoint,
    out (Vector2) secondClosesetIntersectionPoint)
```

Get the angle of a triangle
```cs
flaot angle = SkeletonMath.GetAngle((Vector2) aPoint, (Vector2) bPoint, (Vector2) cPoint);
```

Check if a point is inside a polygon refer to more info about the algorithm [here](https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/)
```cs
bool isInside = PointInPolygon((Vector2) point, (List<Vector2>) polygon);
```

Check if lines are intersecting
```cs
bool areIntersecting = SkeletonMath.GetIntersectionStatus((Vector2)firstLineStartingPoint, (Vector2)firstLineEndingPoint, (Vector2)secondLineStartingPoint, (Vector2)secondLineEndingPoint)
```

Sort a list of polygons by creating a hypothetical straight line
and sorting it from the start to the end of that line
```cs
List<List<Vector2>> sorteredPolygons = SkeletonMath.GetOrderedInners((List<List<Vector2>>)ListOfMyPolygons);
```

Check if a polygon is clockwise or counter
```cs
bool isMyPolygonClockwise = SkeletonMath.IsClockwise((List<Vector2>) myPolygon);
```

Reverse a polygon or a list of any type from clockwise to counterclockwise or vice versa
```cs
List<T> reversedList = SkeletonMath.ReverseList(myList)
```

Get the furthest point on a line from another point
```cs
int indexOfTheFurthestPoint = SkeletonMath.GetFurthestPointIndex((Vector2)pointForMessuring, (List<Vector2>)myListOfPoints);
```

Get the 4 main edges in a polygon
```cs
List<Vector2> myFourCornerPolygon = SkeletonMath.GetQuadrantEdges((List<Vector2>) myPolygon);
```

Get all the edges in a polygon
```cs
List<Vector2> myEdges = SkeletonMath.GetEdges((List<Vector2>)myPolygon,
    (int)segment = 15, // optional! the lower the more polygons
    (bool) processSimplification = false, // optional! significantly reduces the number of polygons if true
    (float) tolerance = 2f, // optional! represents the tolerance of what angle is considered an edge the higher the higher polycount
    (float) bigLineSize = 50 // optional! represents the minimum size of the lines in the polygon 
    );
```

Centroid function 
```cs
Vector2 centroid = SkeletonMath.GetCentroid((List<Vector2>) myPolygon);
```

Simple and very computationally light function for polygon optimization
```cs
List<Vector2> optimizedPolygon = SkeletonMath.GetSimplePolyOptimization((List<Vector2>) myPolygon);
```

Moves a point closer to another pooint
```cs
MovePointTowards(Vector2 PointIWannaMove, Vector2 SecondPoint, double distance);
```
