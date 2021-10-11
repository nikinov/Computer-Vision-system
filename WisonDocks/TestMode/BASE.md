# Test mode docks
To get started with your first test mode you need to take a few important steps few steps you need to do to get your first test mode working.
1. Create a new project in the `CSharpSource\TestModes` folder in a Wision solution
   IMAGE HERE
2. Create the following scripts and derive base classes
3. Add your test mode to the test mode manager
4. create a basic app to get familiar with the features that Wision has to offer
5. explore our documentation to make and ship your first fully functional test mode
6. write your own test mode modules and make it part of our growing documentation  
## PolygonF

### Fixture
Fixture is dedicated to adjusting the polygon to the offset of the image

```CS
var fixtureTransformation = testfield.Polygon.GetFixtureTransformation(segmentKey, dieKey);
```
