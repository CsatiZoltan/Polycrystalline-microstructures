# -*- coding: utf-8 -*-

"""
This module contains functions to provide geometrical descriptions of a label image.
Each labelled region is turned to a surface. This allows mesh generators to work on
the geometry instead of an image, thereby creating good quality meshes.

The following functions require PythonOCC version 0.18 to be installed to allow spline
manipulations:

- `fit_spline`
- `branches2splines`
- `region_as_splinegon`
- `write_step_file`
- `regions2step`
- `plot_splinegons`
- `splinegonize`

Functions
---------
.. autosummary::
    :toctree: functions/

    build_skeleton
    skeleton2regions
    polygon_orientation
    branches2boundary
    region_as_polygon
    polygonize
    branches2splines
    fit_spline
    region_as_splinegon
    splinegonize
    regions2step
    plot_polygons
    plot_splinegons
    write_step_file
    plot_polygon
    overlay_regions
    search_neighbor


"""

import os
import warnings
from collections import Counter
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from skimage.color import label2rgb
from skan import Skeleton, summarize
from skan.draw import overlay_skeleton_networkx

try:
    from OCC.gp import gp_Pnt
    from OCC.TColgp import TColgp_Array1OfPnt
    from OCC.GeomAPI import GeomAPI_PointsToBSpline
    from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, \
        BRepBuilderAPI_MakeWire
    from OCC.TopoDS import TopoDS_Compound
    from OCC.BRep import BRep_Builder
    from OCC.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Interface import Interface_Static_SetCVal
    from OCC.IFSelect import IFSelect_RetDone
    from OCC.Display.SimpleGui import init_display
    from OCC.Display.OCCViewer import rgb_color
except:
    warnings.warn('PythonOCC is not available. Some functions cannot be used.', ImportWarning)

from .utils import toggle, index_list, non_unique, neighborhood


def build_skeleton(label_image, connectivity=1, detect_boundaries=True):
    """Builds skeleton connectivity of a label image.

    A single-pixel wide network is created, separating the labelled image regions. The resulting
    network contains information about how the regions are connected.

    Parameters
    ----------
    label_image : 2D ndarray with signed integer entries
        Label image, representing a segmented image.
    connectivity : {1,2}, optional
        A connectivity of 1 (default) means pixels sharing an edge will be considered neighbors.
        A connectivity of 2 means pixels sharing a corner will be considered neighbors.
    detect_boundaries : bool, optional
        When True, the image boundaries will be treated as part of the skeleton. This allows
        identifying boundary regions in the `skeleton2regions` function. The default is True.

    Returns
    -------
    skeleton_network : Skeleton
        Geometrical and topological information about the skeleton network of the input image.

    See Also
    --------
    :class:`skan.csr.Skeleton`

    """
    # 2D image, given as a numpy array is expected
    if type(label_image) != np.ndarray:
        raise Exception('Label image must be a numpy array (ndarray).')
    image_size = np.shape(label_image)
    if len(image_size) != 2:
        raise Exception('A 2D array is expected.')
    if not issubclass(label_image.dtype.type, np.signedinteger):
        raise Exception('Matrix entries must be positive integers.')

    # Surround the image with an outer region
    if detect_boundaries:
        label_image = np.pad(label_image, pad_width=1, mode='constant', constant_values=-1)

    # Find the boundaries of the label image and then extract its skeleton
    boundaries = find_boundaries(label_image, connectivity=connectivity)
    skeleton = skeletonize(boundaries)

    # Build the skeleton network using `skan`
    skeleton_network = Skeleton(skeleton, source_image=label_image, keep_images=True)
    return skeleton_network


def skeleton2regions(skeleton_network, neighbor_search_algorithm):
    """Determines the regions bounded by a skeleton network.

    This function can be perceived as an intermediate step between a skeleton network and
    completely geometrical representation of the regions. That is, it keeps the key topological
    information required to create a fully geometrical description, but it also contains the
    coordinates of the region boundaries. The outputs of this function can be used to build
    different region representations.

    Parameters
    ----------
    skeleton_network : Skeleton
        Geometrical and topological information about the skeleton network of a label image.
    neighbor_search_algorithm : functools.partial
        Specifies which algorithm to use for constructing the branch-region connectivity.
        The function to be passed (along with its arguments) is :func:`search_neighbor`.
        For further details, see the Notes below.

    Returns
    -------
    region_branches : dict
        For each region it contains the branch indices that bound that region.
    branch_coordinates : list
        Coordinates of the points on each branch.
    branch_regions : dict
        For each branch it contains the neighboring regions.
        This auxiliary data is not essential as it can be restored from :code:`region_branches`.
        However, it is computed as temporary data needed for :code:`region_branches`.

    See Also
    --------
    build_skeleton
    search_neighbor
    overlay_regions

    Notes
    -----
    Although the algorithms were created to require minimum user intervention, some parameters
    must be fine-tuned so as to achieve an optimal result in identifying the regions. Visualization
    plays an important role in it. Full automation is either not possible or would require a huge
    computational cost. The shortcomings of the algorithms in this function are the following:

    - It is assumed that only branches that connect two junctions form the boundary of a region.
      This rule ensures that end points are not taken into account. However, this assumption also
      rules out the identification of regions being contained in another regions as the embedded
      region would be described by an isolated cycle.
    - The recognition of which branches form a region is based on the premise that a node of a
      branch belongs to a region if its `n`-pixel neighbourhood contains a pixel from that region.
      Ideally, `n=1` would be used, meaning that the single-pixel width skeleton is located at most
      1 pixel afar from the regions it lies among. This is true but the nodes of the skeleton
      can be farther than 1 pixel from a region. Hence, `n` has to be a parameter of our model.
      Increasing `n` helps in identifying the connecting regions to a node of a branch.
      On the other hand, if `n` is too large, regions that "in reality" are not neighbors of a
      branch will be included. Currently, we recommend trying different parameters `n`, plot the
      reconstructed regions over the label image using the :func:`overlay_regions` function,
      and see how good the result is. As a heuristic, start with `n=2`.

    """
    if not isinstance(skeleton_network, Skeleton):
        raise Exception('Skeleton object is expected.')
    S = skeleton_network
    skeleton_data = summarize(S)
    mask = skeleton_data['branch-type'] == 2  # only junction-to-junction connections create regions
    image_size = np.shape(S.source_image)
    image_index_grid = [(0, image_size[0] - 1), (0, image_size[1] - 1)]

    # Find which two regions are incident to a branch
    branch_coordinates = [S.path_coordinates(i) for i in range(S.n_paths) if mask[i]]
    branch_regions = []
    for nodes in branch_coordinates:
        c = Counter()
        internal_nodes = range(1, np.size(nodes, 0) - 1)
        for node in internal_nodes:
            # Snap node to the nearest image coordinate
            node_coord = np.round(nodes[node, :]).astype(np.uint32)
            # Look-around for the neighboring pixels (be careful on the image boundaries)
            neighbors = neighbor_search_algorithm(node_coord, bounds=image_index_grid)
            neighbors = S.source_image[neighbors]
            c.update(neighbors)
        neighboring_regions = [pair[0] for pair in c.most_common(2)]
        branch_regions.append(neighboring_regions)
    branch_regions = {key: val for key, val in enumerate(branch_regions)}

    # For each region, find the branches that bound it
    region_branches = {}
    for branch, regions in branch_regions.items():
        for region in regions:
            if region not in region_branches:
                region_branches[region] = [branch]
            else:
                region_branches[region].append(branch)

    return region_branches, branch_coordinates, branch_regions


def polygon_orientation(polygon):
    """Determines whether a polygon is oriented clockwise or counterclockwise.

    Parameters
    ----------
    polygon : list
        Each element of the list denotes a vertex of the polygon and in turn is another list of two
        elements: the x and y coordinates of a vertex.

    Returns
    -------
    orientation : {'cw', 'ccw'}
        'cw': clockwise, 'ccw': counterclockwise orientation

    Notes
    -----
    The formula to determine the orientation is from https://stackoverflow.com/a/1165943/4892892.
    For simple polygons (polygons that admit a well-defined interior), a faster algorithm exits, see
    https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon.

    Examples
    --------
    >>> polygon = [[5, 0], [6, 4], [4, 5], [1, 5], [1, 0]]
    >>> polygon_orientation(polygon)
    'ccw'

    """
    n_vertex = len(polygon)
    edge_sum = 0
    for idx, vertex in enumerate(polygon):
        next_vertex = polygon[(idx + 1) % n_vertex]  # allow indexing past the last vertex
        edge_sum += (next_vertex[0] - vertex[0]) * (next_vertex[1] + vertex[1])
    return 'cw' if edge_sum > 0 else 'ccw'


def branches2boundary(branches, orientation='ccw'):
    """Joins connecting branches so that they form the boundary of a surface.

    This function assumes that you already know that a set of branches form the boundary of a
    surface, but you want to know the ordering. Clockwise or counterclockwise orientation is
    supported. A branch is given by its two end points but it can also contain intermediate points
    in between, as in the general case. The points on the branches are assumed to be ordered. If
    certain branches are not used in forming the boundary of a surface, they are excluded from the
    output lists.

    Parameters
    ----------
    branches : list
        Each element of the list gives N>=2 points on the branch, ordered from one end point to the
        other. If N=2, the two end points are meant. The points are provided as an Nx2 ndarray,
        the first column giving the x, the second column giving the y coordinates of the points.
    orientation : {'cw', 'ccw'}, optional
        Clockwise ('cw') or counterclockwise ('ccw') orientation of the surface boundary.
        The default is 'ccw'.

    Returns
    -------
    order : list
        Order of the branches so that they form the boundary of a surface.
    redirected : list
        A list of bool with True value if the orientation of the corresponding branch had to be
        swapped to form the surface boundary.
    polygon : ndarray
        The polygon formed by connecting the points of the branches along the boundary. It is given
        given as an Mx2 ndarray, where M is the number of unique points on the boundary (i.e. only
        one end point is kept for two connecting branches).
        This auxiliary data is not essential as it can be restored from the set of branches,
        and their ordering. However, it is computed as temporary data needed for determining the
        orientation of the boundary.

    Examples
    --------
    >>> import numpy as np
    >>> branches = [np.array([[1, 1], [1.5, 2], [2, 3]]), np.array([[1, 1], [-1, 2]]),
    ... np.array([[1.5, -3], [2, 3]]), np.array([[1.5, -3], [-1, 2]])]
    >>> order, redirected, polygon = branches2boundary(branches, orientation='cw')
    >>> order
    [0, 2, 3, 1]
    >>> redirected
    [False, True, True, False]
    >>> polygon
    array([[ 1. ,  1. ],
           [ 1.5,  2. ],
           [ 2. ,  3. ],
           [ 1.5, -3. ],
           [-1. ,  2. ]])

    """
    # The path is the collection of consecutively added branches. When there are no more branches
    # to add, the path becomes the boundary of a surface.
    # First, we identify the branches that form the boundary.
    n_branch = len(branches)
    redirected = [False for i in range(n_branch)]
    # Start with an arbitrary branch, the first one
    last_branch = 0  # index of the lastly added branch to the path
    order = [0]
    first_vertex = branches[last_branch][0, :]
    last_vertex = branches[last_branch][-1, :]
    # Visit all vertices of the would-be surface and find the chain of connecting branches
    for _ in range(n_branch - 1):
        if np.allclose(last_vertex, first_vertex):  # one complete cycle is finished
            break
        # Search for branches connecting to the last vertex of the path
        for branch_index, branch in enumerate(branches):
            if branch_index == last_branch:  # exclude the last branch of the path
                continue
            # Check if the first or second end point of the branch connects to the last vertex
            first_endpoint = branch[0, :]
            second_endpoint = branch[-1, :]
            vertex_connects_to_first_endpoint = np.allclose(first_endpoint, last_vertex)
            vertex_connects_to_second_endpoint = np.allclose(second_endpoint, last_vertex)
            if vertex_connects_to_first_endpoint or vertex_connects_to_second_endpoint:
                order.append(branch_index)
                last_branch = branch_index
            if vertex_connects_to_first_endpoint:
                last_vertex = second_endpoint
                break
            elif vertex_connects_to_second_endpoint:
                last_vertex = first_endpoint
                redirected[branch_index] = True  # change the orientation of the connecting branch
                break
            # TODO: add checks against edge cases

    # Polygonize the boundary branches so as to determine the orientation in the next step
    polygon = []
    for branch_index in order:
        branch = branches[branch_index].copy()
        if redirected[branch_index]:
            branch = np.flipud(branch)
        polygon.append(branch[0:-1])
    polygon = np.vstack(polygon)

    # Handle the orientation of the polygon
    if polygon_orientation(polygon) != orientation:
        order.reverse()
        redirected = toggle(redirected)
        polygon = np.flipud(polygon)

    return order, redirected, polygon


def region_as_polygon(branches, orientation='ccw', close='False'):
    """Represents a region as a polygon.

    Based on the combined topological-geometrical (intermediate) representation of the regions,
    (done by :func:`skeleton2regions`), this function provides a fully geometrical description
    of a region.

    Parameters
    ----------
    branches : list
        Each element of the list gives N>=2 points on the branch, ordered from one end point to the
        other. If N=2, the two end points are meant. The points are provided as an Nx2 ndarray,
        the first column giving the x, the second column giving the y coordinates of the points.
        The branches are not necessarily ordered.
    orientation : {'cw', 'ccw'}, optional
        Clockwise ('cw') or counterclockwise ('ccw') orientation of the polygons.
        The default is 'ccw'.
    close : bool, optional
        When True, one vertex in the polygons is repeated to indicate that the polygons are
        indeed closed. The default is False.

    Returns
    -------
    polygon : ndarray
        The resulting polygon, given as an Mx2 ndarray, where M is the number of unique points on
        the polygon (i.e. only one end point is kept for two connecting branches).

    See Also
    --------
    skeleton2regions

    """
    _, _, polygon = branches2boundary(branches, orientation)
    if close:
        polygon = np.vstack((polygon, polygon[0, :]))
    return polygon


def polygonize(label_image, neighbor_search_algorithm, connectivity=1, detect_boundaries=True,
               orientation='ccw', close=False):
    """Polygon representation of a label image.

    Parameters
    ----------
    label_image : 2D ndarray with signed integer entries
        Label image, representing a segmented image.
    neighbor_search_algorithm : functools.partial
        Specifies which algorithm to use for constructing the branch-region connectivity.
        The function to be passed (along with its arguments) is :func:`skeleton2regions`.
    connectivity : {1,2}, optional
        A connectivity of 1 (default) means pixels sharing an edge will be considered neighbors.
        A connectivity of 2 means pixels sharing a corner will be considered neighbors.
    detect_boundaries : bool, optional
        When True, the image boundaries will be treated as part of the skeleton. This allows
        identifying boundary regions in the `skeleton2regions` function. The default is True.
    orientation : {'cw', 'ccw'}, optional
        Clockwise ('cw') or counterclockwise ('ccw') orientation of the polygons.
        The default is 'ccw'.
    close : bool, optional
        When True, one vertex in the polygons is repeated to indicate that the polygons are
        indeed closed. The default is False.

    Returns
    -------
    polygons : dict
        The keys in the dictionary correspond to the labels of the input image, while the values
        are ndarray objects with two columns, the x and y coordinates of the polygons.

    See Also
    --------
    build_skeleton
    skeleton2regions
    region_as_polygon

    Examples
    --------
    >>> test_image = np.array([
    ...   [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
    ...  dtype=np.int8)
    >>> polygons = polygonize(test_image, search_neighbor(2, np.inf), connectivity=1)

    """
    polygons = {}
    # Build the skeleton network from the label image
    S = build_skeleton(label_image, connectivity, detect_boundaries)
    # Identify the regions from the skeleton
    region_branches, branch_coordinates, _ = skeleton2regions(S, neighbor_search_algorithm)
    # Represent each region as a polygon
    for region, branches in region_branches.items():
        if region == -1:  # artificial outer region
            continue
        points_on_boundary = index_list(branch_coordinates, branches)
        poly = region_as_polygon(points_on_boundary, orientation, close)
        polygons[region] = poly
    # overlay_skeleton_networkx(S.graph, S.coordinates, image=label_image)
    return polygons


def branches2splines(branches, degree_min=3, degree_max=8, continuity='C2', tol=1e-3):
    """Lays splines on branches.

    Parameters
    ----------
    branches : list
        Each element of the list gives N>=2 points on the branch, ordered from one end point to the
        other. If N=2, the two end points are meant. The points are provided as an Nx2 ndarray,
        the first column giving the x, the second column giving the y coordinates of the points.

    Other Parameters
    ----------------
    degree_min, degree_max, continuity, tol
        See the :func:`fit_spline` function.

    Returns
    -------
    list
        Each member of the list is a `Geom_BSplineCurve` object, the B-spline approximation of
        the corresponding input branch.
        For details on the resulting spline, see the OpenCASCADE documentation:
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_geom___b_spline_curve.html

    See Also
    --------
    fit_spline

    """
    return [fit_spline(branch, degree_min, degree_max, continuity, tol) for branch in branches]


def fit_spline(points, degree_min=3, degree_max=8, continuity='C2', tol=1e-3):
    """Approximates a set of points on the plane with a B-spline.

    Parameters
    ----------
    points : ndarray
        2D ndarray with N>=2 rows and 2 columns, representing the points on the plane, ordered
        from one end point to the other. The first column gives the x, the second column gives the
        the y coordinates of the points.
    degree_min : int, optional
        Minimum degree of the spline. The default is 3.
    degree_max : int, optional
        Maximum degree of the spline. The default is 8.
    continuity : {'C0', 'G1', 'C1', 'G2', 'C2', 'C3', 'CN'}, optional
        The continuity of the spline will be at least `continuity`. The default is 'C2'.
        For their meanings, consult with
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/_geom_abs___shape_8hxx.html
    tol : float, optional
        The distance from the points to the spline will be lower than `tol`. The default is 1e-3.

    Returns
    -------
    spline : Geom_BSplineCurve
        For details on the resulting spline, see the OpenCASCADE documentation:
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_geom___b_spline_curve.html

    Raises
    ------
    ValueError
        If the minimum degree of the B-spline to be constructed is greater than its maximum degree.

    """
    if degree_min > degree_max:
        raise ValueError('Minimum degree must not be lower than the maximum degree.')
    # Create points from the input array that OCCT understands
    n_point = np.size(points, 0)
    pts = TColgp_Array1OfPnt(0, n_point-1)
    for i in range(n_point):
        pts.SetValue(i, gp_Pnt(float(points[i][0]), float(points[i][1]), 0))
    # Construct the spline
    continuity = _spline_continuity_enum(continuity)
    return GeomAPI_PointsToBSpline(
        pts, degree_min, degree_max, continuity, tol
    ).Curve()


def region_as_splinegon(boundary_splines):
    """Represents a region as a splinegon.

    Based on the combined topological-geometrical (intermediate) representation of the regions,
    (done by :func:`skeleton2regions`), this function provides a fully geometrical description
    of a region.

    Parameters
    ----------
    boundary_splines : list
        Each element of the list is a `Handle_Geom_BSplineCurve` object, giving a reference to the
        B-splines bounding the region. The splines must either be ordered (see
        :func:`branches2boundary`) or they must appear in an order such that the n-th spline in
        the list can be connected to one of the first n-1 splines in the list.

    Returns
    -------
    splinegon : TopoDS_Face
        The resulting splinegon (surface). For details on the object, see the OpenCASCADE API:
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_topo_d_s___face.html
    boundary : TopoDS_Wire
        The boundary of the splinegon. For details on the resulting object, see the OpenCASCADE API:
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_topo_d_s___wire.html

    See Also
    --------
    skeleton2regions
    fit_spline

    Notes
    -----
    The syntax `Handle_classname` in PythonOCC corresponds to wrapping the object of class
    `classname` with a smart pointer. In the C++ interface, it is done by a template:
    `Handle<classname>`.

    """
    # Combine the splines so that they form the boundary (a collection of joining splines)
    boundary = BRepBuilderAPI_MakeWire()
    for spline in boundary_splines:
        edge = BRepBuilderAPI_MakeEdge(spline).Edge()
        boundary.Add(edge)
        _wire_error(boundary.Error())
    # The planar surface (face) is stretched by the wire
    splinegon = BRepBuilderAPI_MakeFace(boundary.Wire())
    _face_error(splinegon.Error())
    return splinegon.Face(), boundary.Wire()


def splinegonize(label_image, neighbor_search_algorithm, connectivity=1, detect_boundaries=True,
                 degree_min=3, degree_max=8, continuity='C2', tol=1e-4):
    """Polygon representation of a label image.

    Parameters
    ----------
    label_image : 2D ndarray with signed integer entries
        Label image, representing a segmented image.
    neighbor_search_algorithm : functools.partial
        Specifies which algorithm to use for constructing the branch-region connectivity.
        The function to be passed (along with its arguments) is :func:`skeleton2regions`.
    connectivity : {1,2}, optional
        A connectivity of 1 (default) means pixels sharing an edge will be considered neighbors.
        A connectivity of 2 means pixels sharing a corner will be considered neighbors.
    detect_boundaries : bool, optional
        When True, the image boundaries will be treated as part of the skeleton. This allows
        identifying boundary regions in the `skeleton2regions` function. The default is True.

    Other Parameters
    ----------------
    degree_min, degree_max, continuity, tol
        See the :func:`fit_spline` function.

    Returns
    -------
    splinegons : dict
        The keys in the dictionary correspond to the labels of the input image, while the values
        are `TopoDS_Face` objects, the surfaces of the regions.
    boundaries : dict
        The keys in the dictionary correspond to the labels of the input image, while the values
        are `TopoDS_Wire` objects, the boundaries of the regions.

    See Also
    --------
    build_skeleton
    skeleton2regions
    region_as_splinegon

    Examples
    --------
    >>> test_image = np.array([
    ...   [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ...   [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
    ...  dtype=np.int8)
    >>> splinegons, _ = splinegonize(test_image, search_neighbor(2, np.inf), connectivity=1, tol=0.1)
    >>> plot_splinegons(list(splinegons.values()))

    """

    # Build the skeleton network from the label image
    S = build_skeleton(label_image, connectivity, detect_boundaries)
    # Identify the regions from the skeleton
    region_branches, branch_coordinates, branch_regions = skeleton2regions(S, neighbor_search_algorithm)
    # Approximate each branch with a B-spline
    splines = branches2splines(branch_coordinates, degree_min, degree_max, continuity, tol)
    # Represent each region as a splinegon
    splinegons = {}
    boundaries = {}
    for region, branches in region_branches.items():
        if region == -1:  # artificial outer region
            continue
        # Splines that form the boundary of the current region
        boundary_splines = index_list(splines, branches)
        # Orient the boundary so that the region surface can be defined
        order, _, _, = branches2boundary(index_list(branch_coordinates, branches))
        # Construct the surface and the boundary of the region
        splinegon, boundary = region_as_splinegon(index_list(boundary_splines, order))
        # Save the created regions and their boundaries
        splinegons[region] = splinegon
        boundaries[region] = boundary
    return splinegons, boundaries


def regions2step(splinegons, filename, application_protocol='AP203'):
    """Writes splinegon regions to a STEP file.

    Parameters
    ----------
    splinegons : list
        Each member of the list is a `TopoDS_Face` object, the surface of a region.
    filename : str
        Name of the file the regions are saved into.
    application_protocol : {'AP203', 'AP214IS', 'AP242DIS'}, optional
        Version of schema used for the output STEP file. The default is 'AP203'.

    Returns
    -------
    None

    See Also
    --------
    splinegonize
    write_step_file

    """
    # Initialize a container that we will populate with the splinegons
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    # Combine multiple faces
    for splinegon in splinegons:
        builder.Add(compound, splinegon)
    # Write to STEP file
    write_step_file(compound, filename, application_protocol)


def plot_polygons(polygons, **kwargs):
    """Plots polygons.

    Parameters
    ----------
    polygons : list
        Each member of the list is a 2D numpy array with 2 columns, each row corresponding to a
        vertex of the polygon, and the two columns giving the Cartesian coordinates of the vertices.

    Other Parameters
    ----------------
    kwargs : optional
        Keyword arguments supported by the constructor of the :class:`matplotlib.figure.Figure`
        class.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure the polygons are plotted into.

    See Also
    --------
    polygonize

    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for poly in polygons:
        plot_polygon(poly, ax)
    return fig


def plot_splinegons(splinegons, transparency=0.5, color='random'):
    """Plots splinegons.

    Parameters
    ----------
    splinegons : list
        Each member of the list is a `TopoDS_Face` object, the surface of a splinegon.
    transparency : float, optional
        The transparency value should be between 0 and 1. At 0, an object will be totally opaque,
        and at 1, fully transparent.
    color : tuple or str, optional
        Color of the splinegons. Either 'random' to associate a random color for each splinegon
        in the list `splinegons` or a 3-tuple with entries between 0 and 1 to give the same RGB
        values to all the splinegons.

    Returns
    -------
    None

    See Also
    --------
    splinegonize

    """
    display, start_display, add_menu, add_function_to_menu = init_display()
    # display.default_drawer.SetFaceBoundaryDraw(False)
    # Check the format of the optional arguments
    if not (0 <= transparency <= 1):
        raise ValueError('Transparency must be in the interval [0, 1].')
    if color != 'random':
        # TODO: Check for format
        col = rgb_color(*color)
    for i, splinegon in enumerate(splinegons):
        # Plot the faces
        if color == 'random':
            col = rgb_color(np.random.random(), np.random.random(), np.random.random())
        display.DisplayShape(splinegon, update=True, transparency=transparency, color=col)
    start_display()


def write_step_file(shape, filename, application_protocol='AP203'):
    """ Exports a shape to a STEP file.

    Parameters
    ----------
    shape : TopoDS_Shape
        Shape to be exported, an object of the `TopoDS_Shape` class or one of its subclasses. See
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/class_topo_d_s___shape.html
    filename : str
        Name of the file the shape is saved into.
    application_protocol : {'AP203', 'AP214IS', 'AP242DIS'}, optional
        Version of schema used for the output STEP file. The default is 'AP203'.

    Notes
    -----
    For details on how to read and write STEP files, see the documentation on
    https://dev.opencascade.org/doc/overview/html/occt_user_guides__step.html.

    """
    if shape.IsNull():
        raise AssertionError('Shape is null.')
    if application_protocol not in ['AP203', 'AP214IS', 'AP242DIS']:
        raise ValueError('Application protocol must be one of {"AP203", "AP214IS", "AP242DIS"}.')
    if os.path.isfile(filename):
        warnings.warn('File already exists and will be replaced.', RuntimeWarning)

    # Create and initialize the STEP exporter
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", application_protocol)

    # Transfer shape and write file
    step_writer.Transfer(shape, STEPControl_AsIs)
    status = step_writer.Write(filename)

    if status != IFSelect_RetDone:
        raise IOError('Error while writing shape to STEP file.')
    if not os.path.isfile(filename):
        raise IOError('File was not saved.')


def plot_polygon(vertices, ax=None, **kwargs):
    """Plots a polygon.

    Parameters
    ----------
    vertices : ndarray
        2D ndarray of size Nx2, with each row designating a vertex and the two columns
        giving the x and y coordinates of the vertices, respectively.
    ax : matplotlib.axes.Axes, optional
        The `Axes` instance the polygon is plotted into. The default is None, in which case a new
        `Axes` within a new figure is created.
    **kwargs : Line2D properties, optional
        Keyword arguments accepted by matplotlib.pyplot.plot

    Returns
    -------
    None

    See Also
    --------
    :func:`matplotlib.pyplot.plot`

    Examples
    --------
    >>> plot_polygon(np.array([[1, 1], [2, 3], [1.5, -3], [-1, 2]]), marker='o');  plt.show()

    """
    # Close the polygon (repeat one vertex) if not yet closed
    first_vertex = vertices[0, :]
    last_vertex = vertices[-1, :]
    closed = np.allclose(first_vertex, last_vertex)
    if not closed:
        vertices = np.vstack((vertices, first_vertex))
    ax = ax or plt.figure().add_subplot()
    ax.plot(vertices[:, 0], vertices[:, 1], **kwargs)


def overlay_regions(label_image, polygons, axes=None):
    """Plots a label image, and overlays polygonal regions over it.

    Parameters
    ----------
    label_image : 2D ndarray with signed integer entries
        Label image, representing a segmented image.
    polygons : dict
        The keys in the dictionary correspond to the labels of the input image, while the values
        are ndarray objects with two columns, the x and y coordinates of the polygons.
        This format is respected by the output of the `polygonize` function.
    axes : matplotlib.axes.Axes, optional
        An Axes object on which to draw. If None, a new one is created.

    Returns
    -------
    axes : matplotlib.axes.Axes
        The Axes object on which the plot is drawn.

    See Also
    --------
    polygonize
    :class:`matplotlib.collections.LineCollection`

    """
    # TODO: support an option to give the number of identified regions as a title
    if axes is None:
        _, axes = plt.subplots()
    # Extract the polygons from the dictionary and convert them to a list to ease the plotting.
    # At the same time, swap the x and y coordinates so that the polygons are expressed in the
    # same coordinate system as the label image.
    polygons = [polygons[i][:, ::-1] for i in polygons.keys()]
    # Plot the polygons efficiently
    axes.add_collection(collections.LineCollection(polygons, colors='black'))
    # Plot the label image, with each color corresponding to a different region
    random_colors = np.random.random((len(polygons), 3))
    plt.imshow(label2rgb(label_image, colors=random_colors))
    # Axis settings
    axes.set_aspect('equal')
    axes.set_axis_off()
    plt.tight_layout()
    return axes


def search_neighbor(radius, norm, method='sphere'):
    """Neighbor search algorithm.

    Parameters
    ----------
    radius, norm, method
        See the :func:`grains.utils.neighborhood` function.

    Returns
    -------
    functools.partial
        A new function with partial application of the given input arguments on the
        :func:`grains.utils.neighborhood` function. The returned partial function is used
        internally by :func:`skeleton2regions` when performing neighbor searching.

    See Also
    --------
    :func:`grains.utils.neighborhood`, :func:`skeleton2regions`

    """
    return partial(neighborhood, radius=radius, norm=norm, method=method)


def _spline_continuity_enum(continuity):
    """Enumeration value corresponding to the continuity of a B-spline.

    Parameters
    ----------
    continuity : {'C0', 'G1', 'C1', 'G2', 'C2', 'C3', 'CN'}
        Continuity of the B-spline.

    Returns
    -------
    enum : int
        Integer value corresponding to continuity key in the OpenCASCADE API:
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/_geom_abs___shape_8hxx.html

    """
    if continuity == 'C0':
        enum = 0
    elif continuity == 'G1':
        enum = 1
    elif continuity == 'C1':
        enum = 2
    elif continuity == 'G2':
        enum = 3
    elif continuity == 'C2':
        enum = 4
    elif continuity == 'C3':
        enum = 5
    elif continuity == 'CN':
        enum = 6
    else:
        raise ValueError('Choose one of {"C0", "G1", "C1", "G2", "C2", "C3", "CN"}.')
    return enum


def _wire_error(error_code):
    """Indicates the outcome of the wire construction.

    Parameters
    ----------
    error_code : int
        Error code returned by the `Error` method of `BRepBuilderAPI_MakeWire`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the error code is 1, 2 or 3. See the OpenCASCADE API:
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/_b_rep_builder_a_p_i___wire_error_8hxx.html
        The exception is raised within this function, as an incompatible wire makes it
        impossible to construct a valid surface.

    """
    if error_code == 0:
        pass  # No error occurred. The wire is correctly built.
    elif error_code == 1:
        raise ValueError('Empty constructor was used. The wire does not yet exist.')
    elif error_code == 2:
        raise ValueError('Disconnected wire. The last edge you attempted to add was not '
                         'connected to the wire.')
    elif error_code == 3:
        raise ValueError('Non-manifold wire. The wire has singularity.')
    else:
        raise ValueError('Error code not recognized. It must be one of the following: 0, 1, 2, 3.')


def _face_error(error_code):
    """Indicates the outcome of the face construction.

    Parameters
    ----------
    error_code : int
        Error code returned by the `Error` method of `BRepBuilderAPI_MakeFace`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the error code is 1, 2, 3 or 4. See the OpenCASCADE API:
        https://www.opencascade.com/doc/occt-7.4.0/refman/html/_b_rep_builder_a_p_i___face_error_8hxx.html
        The exception is raised within this function if the face could not be created.

    """
    if error_code == 0:
        pass  # No error occurred. The face is correctly built.
    elif error_code == 1:
        raise ValueError('Empty constructor was used. The face does not yet exist.')
    elif error_code == 2:
        raise ValueError('The face cannot be constructed from a non-planar wire.')
    elif error_code == 3:
        raise ValueError('Curve projection failed.')
    elif error_code == 4:
        raise ValueError('The parameters given to limit the surface are out of its bounds.')
    else:
        raise ValueError('Error code not recognized. It must be one of the following: 0, 1, 2, 3, 4.')


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
