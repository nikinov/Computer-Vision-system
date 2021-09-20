from matplotlib import pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.ops import cascaded_union


path = "../../../resources/CoordinateData/maNiceOne.txt"
f = open(path)

def plot_lines():
    all_lines = []
    lines = f.read().split("|\n")
    for line in lines:
        l = []
        for ll in line.split("|"):
            try:
                l.append(ll.split(","))
            except:
                pass
        if l is not None:
            all_lines.append(l)

    for lin in all_lines:
        try:
            x_values = [lin[0][0], lin[1][0]]

            y_values = [lin[0][1], lin[1][1]]

            plt.plot(x_values, y_values)
        except:
            pass

def plot_polygon():
    fig,ax = plt.subplots(1)

    new_coords = []

    for line in f.read().split(",\n"):
        try:
            coords = []
            for lin in line.split(","):
                try:
                    li = lin.split("|")
                    coords.append([float(li[0]), float(li[1])])
                except:
                    pass
            coords.append(coords[0])
            new_coords.append(coords)
        except:
            pass

    for cor in new_coords:
        xs, ys = zip(*cor)
        fig = plt.figure()
        plt.plot(xs, ys)

plot_polygon()
plt.show()
