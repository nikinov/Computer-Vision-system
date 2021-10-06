# importing the modules
from bokeh.plotting import figure, output_file, show


# data paths
path = "../../../resources/CoordinateData/maNiceOne.txt"
f = open(path)

# file to save the model
output_file("gfg.html")

# instantiating the figure object
graph = figure(title="Bokeh Multiple Polygons Graph")

# name of the x-axis
graph.xaxis.axis_label = "x-axis"

# name of the y-axis
graph.yaxis.axis_label = "y-axis"

# the points to be plotted
xs = []
ys = []
for line in f.read().split(",\n"):
    try:
        coodrsx = []
        coordsy = []
        for lin in line.split(","):
            try:
                li = lin.split("|")
                coodrsx.append(float(li[0]))
                coordsy.append(float(li[1]))
            except:
                pass

        xs.append(coodrsx)
        ys.append(coordsy)
    except:
        pass

# color values of the poloygons
color = ["red", "purple", "yellow"]

# plotting the graph
graph.multi_polygons(xs, ys)

# displaying the model
show(graph)

