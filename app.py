from flask import Flask, render_template, request, send_file, make_response
from flask_cors import CORS
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import mapping

app = Flask(__name__)
CORS(app)


# ============================

def inpoly2(vert, node, edge=None, ftol=5.0e-14):
    """
    INPOLY2: compute "points-in-polygon" queries.


    STAT, BNDS = INPOLY2(..., FTOL) also returns an N-by-1
    boolean array BNDS, with BNDS[II] = TRUE if VERT[II, :]
    lies "on" a boundary segment, where FTOL is a floating-
    point tolerance for boundary comparisons. By default,
    FTOL ~ EPS ^ 0.85.

    --------------------------------------------------------

    This algorithm is based on a "crossing-number" test,
    counting the number of times a line extending from each
    point past the right-most end of the polygon intersects
    with the polygonal boundary. Points with odd counts are
    "inside". A simple implementation requires that each
    edge intersection be checked for each point, leading to
    O(N*M) complexity...


    """

    vert = np.asarray(vert, dtype=np.float64)
    node = np.asarray(node, dtype=np.float64)

    if edge is None:
#----------------------------------- set edges if not passed
        indx = np.arange(0, node.shape[0] - 1)

        edge = np.zeros((
            node.shape[0], +2), dtype=np.int32)

        edge[:-1, 0] = indx + 0
        edge[:-1, 1] = indx + 1
        edge[ -1, 0] = node.shape[0] - 1

    else:
        edge = np.asarray(edge, dtype=np.int32)

    STAT = np.full(
        vert.shape[0], False, dtype=np.bool_)
    BNDS = np.full(
        vert.shape[0], False, dtype=np.bool_)

#----------------------------------- prune points using bbox
    mask = np.logical_and.reduce((
        vert[:, 0] >= np.nanmin(node[:, 0]),
        vert[:, 1] >= np.nanmin(node[:, 1]),
        vert[:, 0] <= np.nanmax(node[:, 0]),
        vert[:, 1] <= np.nanmax(node[:, 1]))
    )

    vert = vert[mask]

#------------------ flip to ensure y-axis is the `long` axis
    xdel = np.amax(vert[:, 0]) - np.amin(vert[:, 0])
    ydel = np.amax(vert[:, 1]) - np.amin(vert[:, 1])

    lbar = (xdel + ydel) / 2.0

    if (xdel > ydel):
        vert = vert[:, (1, 0)]
        node = node[:, (1, 0)]

#----------------------------------- sort points via y-value
    swap = node[edge[:, 1], 1] < node[edge[:, 0], 1]
    temp = edge[swap]
    edge[swap, :] = temp[:, (1, 0)]

#----------------------------------- call crossing-no kernel
    stat, bnds = \
        _inpoly(vert, node, edge, ftol, lbar)

#----------------------------------- unpack array reindexing
    STAT[mask] = stat
    BNDS[mask] = bnds

    return STAT, BNDS


def _inpoly(vert, node, edge, ftol, lbar):
    """
    _INPOLY: the local pycode version of the crossing-number
    test. Loop over edges; do a binary-search for the first
    vertex that intersects with the edge y-range; crossing-
    number comparisons; break when the local y-range is met.

    """

    feps = ftol * (lbar ** +2)
    veps = ftol * (lbar ** +1)

    stat = np.full(
        vert.shape[0], False, dtype=np.bool_)
    bnds = np.full(
        vert.shape[0], False, dtype=np.bool_)

#----------------------------------- compute y-range overlap
    ivec = np.argsort(vert[:, 1], kind="quicksort")

    XONE = node[edge[:, 0], 0]
    XTWO = node[edge[:, 1], 0]
    YONE = node[edge[:, 0], 1]
    YTWO = node[edge[:, 1], 1]

    XMIN = np.minimum(XONE, XTWO)
    XMAX = np.maximum(XONE, XTWO)

    XMAX = XMAX + veps
    YMIN = YONE - veps
    YMAX = YTWO + veps

    YDEL = YTWO - YONE
    XDEL = XTWO - XONE

    ione = np.searchsorted(
        vert[:, 1], YMIN,  "left", sorter=ivec)
    itwo = np.searchsorted(
        vert[:, 1], YMAX, "right", sorter=ivec)

#----------------------------------- loop over polygon edges
    for epos in range(edge.shape[0]):

        xone = XONE[epos]; xtwo = XTWO[epos]
        yone = YONE[epos]; ytwo = YTWO[epos]

        xmin = XMIN[epos]; xmax = XMAX[epos]

        xdel = XDEL[epos]; ydel = YDEL[epos]

    #------------------------------- calc. edge-intersection
        for jpos in range(ione[epos], itwo[epos]):

            jvrt = ivec[jpos]

            if bnds[jvrt]: continue

            xpos = vert[jvrt, 0]
            ypos = vert[jvrt, 1]

            if xpos >= xmin:
                if xpos <= xmax:
                #------------------- compute crossing number
                    mul1 = ydel * (xpos - xone)
                    mul2 = xdel * (ypos - yone)

                    if feps >= abs(mul2 - mul1):
                #------------------- BNDS -- approx. on edge
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == yone) and (xpos == xone):
                #------------------- BNDS -- match about ONE
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == ytwo) and (xpos == xtwo):
                #------------------- BNDS -- match about TWO
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (mul1 <= mul2) and (ypos >= yone) \
                            and (ypos < ytwo):
                #------------------- advance crossing number
                        stat[jvrt] = not stat[jvrt]

            elif (ypos >= yone) and (ypos < ytwo):
            #----------------------- advance crossing number
                stat[jvrt] = not stat[jvrt]

    return stat, bnds


try:
#-- automagically "override" _inpoly with a compiled kernel!
    from inpoly.inpoly_ import _inpoly  # noqa

except ImportError:
#-- if it hasn't been built, just stick with the .py version
    pass

# ==============================================

def getNodeGeojson(polyName):
    os.system("pwd")
    gdf = gpd.read_file("stanford-sh819zz8121-geojson.json")
    # print(gdf)

    gseries = gdf[gdf['laa'] == polyName]['geometry'] # GeoSeries (Geopandas equivalent of Series from pandas) object

    polygon = gseries[gseries.keys()[0]][0]  # Retrieve Polygon object
    polygon = np.asarray(mapping(polygon)['coordinates'])  # Convert from Polygon into numpy array
    polygon = polygon.reshape(-1, 3)  # Remove extra dimension in array
    polygon = np.delete(polygon, 2, axis=1)  # Remove z-axis (all 0s)

    # print(polygon)
    # print(type(polygon))
    # print(polygon.shape)

    return polygon


# ============================


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/poly', methods=['GET'])
def polygon():
    # poly_name = request.get_json()['laa'] # name of polygon state

    poly_name = request.args.get('name')  # name of polygon state
    print(poly_name)

    # Get polygon from district (LAA) name -- see util function
    node = getNodeGeojson(poly_name)

    # Create grid of points within bounding box that encompasses India (note that a lot of points will fall out of
    # India itself, but this is just for purposes of this demo

    xpos, ypos = np.meshgrid(
        np.linspace(68, 98, 316), np.linspace(8, 40, 316))

    points = np.concatenate((
        np.reshape(xpos, (xpos.size, 1)),
        np.reshape(ypos, (ypos.size, 1))), axis=1)

    # print(points)

    np.set_printoptions(threshold=1000)

    # print(node)
    IN, ON = inpoly2(points, node)

    # Get index (1 to 100000) for all coordinates (people) within polygon
    points_df = pd.DataFrame(points)
    indices = points_df[(IN == 1) | (ON == 1)].index

    # Read synthetic population and select relevant indices to form subpopulation (people w/in polygon)
    synthpop = pd.read_csv("synth100k.csv")
    subpopulation = synthpop.iloc[indices]

    print(subpopulation)

    # Send subpop as downloadable csv
    resp = make_response(subpopulation.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    resp.mimetype = 'text/csv'
    return resp


@app.route('/polyage', methods=['GET'])
def polygonage():
    # poly_name = request.get_json()['laa'] # name of polygon state

    poly_name = request.args.get('name')  # name of polygon state
    print(poly_name)

    # Get polygon from district (LAA) name -- see util function
    node = getNodeGeojson(poly_name)

    # Create grid of points within bounding box that encompasses India (note that a lot of points will fall out of
    # India itself, but this is just for purposes of this demo

    xpos, ypos = np.meshgrid(
        np.linspace(68, 98, 316), np.linspace(8, 40, 316))

    points = np.concatenate((
        np.reshape(xpos, (xpos.size, 1)),
        np.reshape(ypos, (ypos.size, 1))), axis=1)

    # print(points)

    np.set_printoptions(threshold=1000)

    # print(node)
    IN, ON = inpoly2(points, node)

    # Get index (1 to 100000) for all coordinates (people) within polygon
    points_df = pd.DataFrame(points)
    indices = points_df[(IN == 1) | (ON == 1)].index

    # Read synthetic population and select relevant indices to form subpopulation (people w/in polygon)
    synthpop = pd.read_csv("synth100k.csv")
    subpopulation = synthpop.iloc[indices]
    subpopulation = subpopulation.loc[subpopulation["Age"] > 40]

    print(subpopulation)

    # Send subpop as downloadable csv
    resp = make_response(subpopulation.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    resp.mimetype = 'text/csv'
    return resp


@app.route('/polysub', methods=['GET'])
def polygonsub():
    # poly_name = request.get_json()['laa'] # name of polygon state

    poly_name = request.args.get('name')  # name of polygon state
    print(poly_name)

    # Get polygon from district (LAA) name -- see util function
    node = getNodeGeojson(poly_name)

    # Create grid of points within bounding box that encompasses India (note that a lot of points will fall out of
    # India itself, but this is just for purposes of this demo

    xpos, ypos = np.meshgrid(
        np.linspace(68, 98, 316), np.linspace(8, 40, 316))

    points = np.concatenate((
        np.reshape(xpos, (xpos.size, 1)),
        np.reshape(ypos, (ypos.size, 1))), axis=1)

    # print(points)

    np.set_printoptions(threshold=1000)

    # print(node)
    IN, ON = inpoly2(points, node)

    # Get index (1 to 100000) for all coordinates (people) within polygon
    points_df = pd.DataFrame(points)
    indices = points_df[(IN == 1) | (ON == 1)].index

    # Read synthetic population and select relevant indices to form subpopulation (people w/in polygon)
    synthpop = pd.read_csv("synth100k.csv")
    subpopulation = synthpop.iloc[indices]
    subpopulation = subpopulation[["Job", "Religion", "Caste"]]

    print(subpopulation)

    # Send subpop as downloadable csv
    resp = make_response(subpopulation.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    resp.mimetype = 'text/csv'
    return resp


@app.route('/getpopulation')
def get_population():
    size = request.args.get("size")

    # Read the correct file by size, so options for size are 100K, 1M, 10M
    df = pd.read_csv("saves/india" + str(size) + ".csv")

    # Takes all &state= args and puts them in a list
    states = request.args.getlist("states")

    # If states are specified, only returns data from those states. Otherwise returns all data in specified csv file
    if states:
        print("Filtering out all states except " + str(states))
        df = df[df['state'].isin(states)]

    # TODO: add filtering by district

    # Download file on user's machine
    df.to_csv("/tmp/dfgen.csv")
    response = make_response(send_file('/tmp/dfgen.csv'), 200)

    print(f"Returning population sample of size {df.size}")
    return response


if __name__ == '__main__':
    app.run()
