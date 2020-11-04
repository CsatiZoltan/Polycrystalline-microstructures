# -*- coding: utf-8 -*-
"""
This module implements computational geometry algorithms, needed for other modules.

All the examples assume that the modules `numpy` and `matplotlib.pyplot` were imported as `np`
and `plt`, respectively.

Classes
-------
.. autosummary::
    :nosignatures:
    :toctree: classes/

    Mesh
    TriMesh
    Polygon

Functions
---------
.. autosummary::
    :toctree: functions/

    is_collinear
    squared_distance
    distance_matrix
    _polygon_area

"""
from abc import ABC, abstractmethod
from math import sqrt, isclose, pi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from grains.utils import parse_kwargs


class Mesh(ABC):
    """Data structure for a general mesh.

    This class by no means wants to provide intricate functionalities and does not strive to be
    efficient at all. Its purpose is to give some useful features the project relies on.
    The two main entities in the mesh are the `vertices` and the `cells`. They are expected to be
    passed by the user, so reading from various mesh files is not implemented. This keeps the
    class simple, requires few package dependencies and keeps the class focused as there are
    powerful tools to convert among mesh formats (see e.g. `meshio
    <https://github.com/nschloe/meshio>`_).

    Parameters
    ----------
    vertices : ndarray
        2D numpy array with 2 columns, each row corresponding to a vertex, and the two columns
        giving the Cartesian coordinates of the vertex.
    cells : ndarray
        Cell-vertex connectivities in a 2D numpy array, in which each row corresponds to a
        cell and the columns are the vertices of the cells. It is assumed that all the cells
        have the same number of vertices.

    See Also
    --------
    ~TriMesh.change_vertex_numbering

    Notes
    -----
    Although not necessary, it is highly recommended that the local vertex numbering in the
    cells are the same, either clockwise or counter-clockwise. Some methods, such as
    :meth:`get_boundary` even requires it. If you are not sure whether the cells you provide
    have a consistent numbering, it is better to renumber them by calling the
    :meth:`~TriMesh.change_vertex_numbering` method.

    """

    # @abstractmethod
    def __init__(self, vertices, cells):
        """
        .. todo::
            Error checking

        """
        self.vertices = vertices
        self.cells = cells
        self.vertex_sets = {}
        self.cell_sets = {}
        self.field = {'name': None, 'values': None}  # field attached to the mesh

    def create_cell_set(self, name, cells):
        """Forms a group from a set of cells.

        Parameters
        ----------
        name : str
            Name of the cell set.
        cells : list
            List of cells to be added to the set.

        Returns
        -------
        None

        """
        self.cell_sets[name] = cells

    def create_vertex_set(self, name, vertices):
        """Forms a group from a set of vertices.

        Parameters
        ----------
        name : str
            Name of the vertex set.
        vertices : list
            List of vertices to be added to the set.

        """
        self.vertex_sets[name] = vertices

    def associate_field(self, vertex_values, name='field'):
        """Associates a scalar, vector or tensor field to the nodes.

        Only one field can be present at a time. If you want to use a new field, call this method
        again with the new field values, which will replace the previous ones.

        Parameters
        ----------
        vertex_values : ndarray
            Field values at the nodes.
        name : str, optional
            Name of the field. If not given, it will be 'field'.

        Returns
        -------
        None

        """
        if Mesh._isvector(vertex_values):  # store as a column vector
            vertex_values = np.reshape(vertex_values, (len(vertex_values), 1))
        elif not Mesh._ismatrix(vertex_values):
            raise ValueError('Vertex values are expected to be given in a vector or in a matrix.')
        self.field = {'name': name, 'values': vertex_values}

    def get_edges(self):
        """Constructs edge-cell connectivities of the mesh.

        The cells of the mesh do not necessarily have to have a consistent vertex numbering.

        Returns
        -------
        edges : dict
            The keys of the returned dictionary are 2-tuples, representing the two vertices of
            the edges, while the values are the list of cells containing a particular edge.

        Notes
        -----
        We traverse through the cells of the mesh, and within each cell the edges. The edges are
        stored as new entries in a dictionary if they are not already stored. Checking if a key
        exists in a dictionary is performed in O(1). The number of edges in a cell is independent
        of the mesh density. Therefore, the time complexity of the algorithm is O(N), where N is
        the number of cells in the mesh.

        See Also
        --------
        get_boundary

        Examples
        --------
        We show an example for a triangular mesh (as the :class:`Mesh` class is abstract).

        >>> mesh = TriMesh(np.array([[0, 0], [1, 0], [2, 0], [0, 2], [0, 1], [1, 1]]),
        ...                np.array([[0, 1, 5], [4, 5, 3], [5, 4, 0], [2, 5, 1]]))
        >>> edges = mesh.get_edges()
        >>> edges  # doctest: +NORMALIZE_WHITESPACE
        {(0, 1): [0], (1, 5): [0, 3], (5, 0): [0, 2], (4, 5): [1, 2], (5, 3): [1],
        (3, 4): [1], (4, 0): [2], (2, 5): [3], (1, 2): [3]}
        >>> mesh.plot(cell_labels=True, vertex_labels=True)
        >>> plt.show()

        """
        edges = {}
        n_cell_edge = np.size(self.cells, 1)
        for cell, cell_edges in enumerate(self.cells):
            for i in np.arange(n_cell_edge):
                edge = tuple(np.roll(cell_edges, -i)[0:2])
                edge_reverse_orientation = edge[::-1]
                if edge in edges:
                    edges[edge].append(cell)
                elif edge_reverse_orientation in edges:
                    edges[edge_reverse_orientation].append(cell)
                else:
                    edges[edge] = [cell]
        if any(len(i) not in {1, 2} for i in edges.values()):
            raise Exception('Each edge must have either 1 or 2 neighboring cells.')
        return edges

    def get_boundary(self):
        """Extracts the boundary of the mesh.

        It is expected that all the cells have the same orientation, i.e. the cell vertices are
        consistently numbered (either clockwise or counter-clockwise). See the constructor for
        details.

        Returns
        -------
        boundary_vertices : ndarray
            Ordered 1D ndarray of vertices, the boundary vertices of the mesh.
        boundary_edges : dict
            The keys of the returned dictionary are 2-tuples, representing the two vertices of
            the boundary edges, while the values are the list of cells containing a particular
            boundary edge. The dictionary is ordered: the consecutive keys represent the
            consecutive boundary edges. Although a boundary edge is part of a single cell, that
            cell is given in a list so as to maintain the same format as the one used in the
            :meth:`get_edges` method.

        Notes
        -----
        The reason why consistent cell vertex numbering is demanded is because in that case the
        boundary edges are oriented in such a way that the second vertex of a boundary edge is
        the first vertex of the boundary edge it connects to.

        Examples
        --------
        Let us consider the same example mesh as the one described in the :meth:`get_edges` method.

        >>> mesh = TriMesh(np.array([[0, 0], [1, 0], [2, 0], [0, 2], [0, 1], [1, 1]]),
        ...                np.array([[0, 1, 5], [4, 5, 3], [5, 4, 0], [2, 5, 1]]))

        We extract the boundary of that mesh using

        >>> bnd_vertices, bnd_edges = mesh.get_boundary()
        >>> bnd_vertices
        array([0, 1, 2, 5, 3, 4])
        >>> bnd_edges
        {(0, 1): [0], (1, 2): [3], (2, 5): [3], (5, 3): [1], (3, 4): [1], (4, 0): [2]}

        """
        # All edges in the mesh
        edges = self.get_edges()
        # Consider only those edges that are on the boundary
        boundary_edges = {edge: cells for edge, cells in edges.items() if len(cells) == 1}
        boundary_edges = np.array(list(boundary_edges.keys()))
        # Preallocate the arrays holding the boundary edges and the boundary vertices
        n_boundary_edge = np.size(boundary_edges, 0)
        n_boundary_vertex = n_boundary_edge + 1
        boundary_edges_interlaced = np.empty((n_boundary_edge, 2), dtype=int)
        boundary_vertices = np.empty(n_boundary_vertex, dtype=int)
        # Start the processing with the first boundary edge and its two vertices
        boundary_edges_interlaced[0, :] = boundary_edges[0, :]
        boundary_vertices[0:2] = boundary_edges_interlaced[0, :]
        # Find the consecutive boundary edges and their vertices
        for i in np.arange(1, n_boundary_edge):
            # The next boundary edge starts with the last boundary vertex we found
            next_boundary_edge_idx = np.where(boundary_edges[:, 0] == boundary_vertices[i])[0]
            next_boundary_edge = boundary_edges[next_boundary_edge_idx, :]
            # The next boundary vertex is the end point of this new boundary edge
            next_boundary_vertex = next_boundary_edge[0, 1]
            # Update the already found consecutive boundary edges and vertices
            boundary_edges_interlaced[i, :] = next_boundary_edge
            boundary_vertices[i+1] = next_boundary_vertex
        # The last vertex is the same as the first one: keep only one
        boundary_vertices = boundary_vertices[:-1]
        # Bring the boundary edges to the same format as the set of all edges
        boundary_edges = {tuple(edge): edges[tuple(edge)] for edge in boundary_edges_interlaced}
        return boundary_vertices, boundary_edges

    @staticmethod
    def _isvector(array):
        """Decides whether the input is a vector.

        Parameters
        ----------
        array : ndarray
            Numpy array to be checked.

        Returns
        -------
        bool
            True if the input is a 1D array or if it is a column or row vector. Otherwise, False.

        See Also
        --------
        _ismatrix

        """
        shape = np.shape(array)
        if len(shape) == 1:
            return True
        elif len(shape) == 2 and shape[0] == 1 or shape[1] == 1:
            return True
        else:
            return False

    @staticmethod
    def _ismatrix(array):
        """Decides whether the input is a matrix.

        Parameters
        ----------
        array : ndarray
            Numpy array to be checked.

        Returns
        -------
        bool
            True if the input is a 2D array. Otherwise, False.

        See Also
        --------
        _isvector

        """
        shape = np.shape(array)
        return len(shape) == 2


class TriMesh(Mesh):
    """Unstructured triangular mesh.

    Vertices and cells are both stored as numpy arrays. This makes the simple mesh manipulations
    easy and provides interoperability with the whole scientific Python stack.

    Parameters
    ----------
    vertices : ndarray
        2D numpy array with 2 columns, each row corresponding to a vertex, and the two columns
        giving the Cartesian coordinates of the vertex.
    cells : ndarray
        Cell-vertex connectivities in a 2D numpy array, in which each row corresponds to a
        cell and the columns are the vertices of the cells. It is assumed that all the cells
        have the same number of vertices.

    """
    # Default settings for plotting the mesh
    plot_options = {'ax': None,  # Axes object the mesh is plotted
                    'cell_sets': True, 'vertex_sets': True,  # plot groups (if any)
                    'cell_legends': False, 'vertex_legends': False,  # show group legends
                    'cell_labels': False, 'vertex_labels': False  # show labels
                    }

    def __init__(self, vertices, cells):
        super().__init__(vertices, cells)

    def cell_set_to_mesh(self, cell_set):
        """Creates a mesh from a cell set.

        The cell orientation is preserved. I.e. if the cells had a consistent orientation
        (clockwise or counter-clockwise), the cells of the new mesh inherit this property.

        Parameters
        ----------
        cell_set : str
            Name of the cell set being used to construct the new mesh. The cell set must be
            present in the :code:`cell_sets` member variable of the current mesh object.

        Returns
        -------
        TriMesh
            A new :class:`TriMesh` object, based on the selected cell set of the original mesh.

        Notes
        -----
        The implementation is based on https://stackoverflow.com/a/13572640/4892892.

        Examples
        --------
        >>> mesh = TriMesh(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
        ...                np.array([[0, 1, 2], [1, 3, 2]]))
        >>> mesh.create_cell_set('set', [1])
        >>> new_mesh = mesh.cell_set_to_mesh('set')
        >>> new_mesh.cells  # note that the vertices have been relabelled
        array([[0, 2, 1]])
        >>> new_mesh.vertices
        array([[1, 0],
               [0, 1],
               [1, 1]])
        >>> new_mesh.plot(cell_labels=True, vertex_labels=True)
        >>> plt.show()

        """
        # Cells and vertices in the given cell set
        cells = self.cells[self.cell_sets[cell_set]]
        vertices = np.unique(cells)
        # Indices where the vertices can be found in the cell-vertex connectivity matrix
        cell_vertex_flattened = np.digitize(cells.ravel(), vertices, right=True)
        # These indices, at the same time, give the new cell labels
        # (because the vertices are renumbered as 0, 1, ...)
        cells_renumbered = cell_vertex_flattened.reshape(cells.shape)
        return TriMesh(self.vertices[vertices], cells_renumbered)

    def change_vertex_numbering(self, orientation, inplace=False):
        """Changes cell vertex numbering.

        Parameters
        ----------
        orientation : {'ccw', 'cw'}
            Vertex numbering within a cell, either `'ccw'` (counter-clockwise, default) or
            `'cw'` (clock-wise).
        inplace : bool, optional
            If True, the vertex ordering is updated in the mesh object. The default is False.

        Returns
        -------
        reordered_cells : ndarray
            Same format as the :code:`cells` member variable, with the requested vertex ordering.

        Notes
        -----
        Supposed to be used with planar P1 or Q1 finite s.

        Examples
        --------
        >>> mesh = TriMesh(np.array([[1, 1], [3, 5], [7,3]]), np.array([0, 1, 2]))
        >>> mesh.change_vertex_numbering('ccw')
        array([[2, 1, 0]])

        """
        if np.ndim(self.cells) == 1:  # single cell, given as a vector
            self.cells = self.cells.reshape((1, np.size(self.cells)))  # reshape to a matrix
        if np.size(self.cells, 1) < 3:
            raise Exception('Cells must have at least 3 vertices.')
        reordered_cells = np.empty_like(self.cells, dtype=int)
        for i, cell_vertices in enumerate(self.cells):
            coordinates = self.vertices[cell_vertices]
            signed_area = _polygon_area(list(coordinates[:, 0]), list(coordinates[:, 1]))
            if (signed_area < 0 and orientation == 'ccw') or \
               (signed_area > 0 and orientation == 'cw'):
                cell_vertices = cell_vertices[::-1]
            reordered_cells[i] = cell_vertices
        if inplace:
            self.cells = reordered_cells
        return reordered_cells

    def scale(self, factor, inplace=False):
        """Scales the geometry by modifying the coordinates of the vertices.

        Parameters
        ----------
        factor : float
            Each vertex coordinate is multiplied by this non-negative number.
        inplace : bool, optional
            If True, the vertex positions are updated in the mesh object. The default is False.

        Returns
        -------
        None.

        """
        scaled_vertices = factor*self.vertices
        if inplace:
            self.vertices = scaled_vertices
        return scaled_vertices

    def rotate(self, angle, point=(0, 0), inplace=False):
        """Rotates a 2D mesh about a given point by a given angle.

        Parameters
        ----------
        angle : float
            Angle of rotation, in radians.
        point : list or tuple, optional
            Coordinates of the point about which the mesh is rotated. If not given, it is the origin.
        inplace : bool, optional
            If True, the vertex positions are updated in the mesh object. The default is False.

        Returns
        -------
        rotated_vertices : ndarray
            Same format as the :code:`vertices` member variable, with the requested rotation.

        Notes
        -----
        Rotating a point :math:`P` given by its coordinates in the global coordinate system as
        :math:`P(x,y)` around a point :math:`A(x,y)` by an angle :math:`\\alpha` is done as follows.

        1. The coordinates of :math:`P` in the local coordinate system, the origin of which is
        :math:`A`, is expressed as

        .. math::
            P(x',y') = P(x,y) - A(x,y).

        2. The rotation is performed in the local coordinate system as :math:`P'(x',y') = RP(x',y')`,
        where :math:`R` is the rotation matrix:

        .. math::
            R = \\begin{pmatrix}
                \\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\hphantom{-}\\cos\\alpha
                \\end{pmatrix}.

        3. The rotated point :math:`P'` is expressed in the original (global) coordinate system:

        .. math::
            P'(x,y) = P'(x',y') + A(x,y).

        """
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_vertices = np.empty_like(self.vertices)
        for i, vertex in enumerate(self.vertices):
            vertex_local = vertex - point
            rotated_vertex_local = rotation_matrix.dot(vertex_local)
            rotated_vertex = rotated_vertex_local + point
            rotated_vertices[i] = rotated_vertex
        if inplace:
            self.vertices = rotated_vertices
        return rotated_vertices

    def cell_area(self, cell):
        """Computes the area of a cell.

        Parameters
        ----------
        cell : int
            Cell label.

        Returns
        -------
        area : float
            Area of the cell.

        See Also
        --------
        cell_set_area

        """
        vertices = self.cells[cell]
        coordinates = self.vertices[vertices]
        return _polygon_area(list(coordinates[:, 0]), list(coordinates[:, 1]))

    def cell_set_area(self, cell_set):
        """Computes the area of a cell set.

        Parameters
        ----------
        cell_set : str
            Name of the cell set.

        Returns
        -------
        area : float
            Area of the cell set.

        See Also
        --------
        cell_area

        """
        return sum(self.cell_area(cell) for cell in self.cell_sets[cell_set])

    def plot(self, *args, **kwargs):
        """Plots the mesh.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The `Axes` instance the mesh resides in. The default is None, in which case a new
            `Axes` within a new figure is created.

        Other Parameters
        ----------------
        cell_sets, vertex_sets : bool, optional
            If True, the cell/vertex sets (if exist) are highlighted in random colors.
            The default is True.
        cell_legends, vertex_legends : bool, optional
            If True, cell/vertex set legends are shown. The default is False. For many sets,
            it is recommended to leave these options as False, otherwise the plotting becomes
            very slow.
        cell_labels, vertex_labels : bool, optional
            If True, cell/vertex labels are shown. The default is False. Recommended to be left
            False in case of many cells/vertices. Cell labels are positioned in the centroids
            of the cells.
        args, kwargs : optional
            Additional arguments and keyword arguments to be specified. Those arguments are the
            ones supported by :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        None

        Notes
        -----
        If you do not want to plot the cells, only the vertices, pass the :code:`'.'` option, e.g.:

        .. code-block:: python

            mesh.plot('k.')

        to plot the vertices in black. Here, :code:`mesh` is a `TriMesh` object.

        Examples
        --------
        A sample mesh is constructed by creating uniformly randomly distributed points on the
        rectangular domain [-1, 1] x [1, 2]. These points will constitute the vertices of the
        mesh, while its cells are the Delaunay triangles on the vertices.

        .. plot::

            >>> from grains.geometry import TriMesh
            >>> msh = TriMesh(*TriMesh.sample_mesh(1))

            The cells are drawn in greeen, in 3 points of line width, and the vertices of the mesh
            are shown in blue.

            >>> msh.plot('go-', linewidth=3, markerfacecolor='b', vertex_labels=True)
            >>> plt.show()

        Notes
        -----
        The plotting is done by calling :func:`~matplotlib.pyplot.triplot`, which internally
        makes a deep copy of the triangles. This increases the memory usage in case of many
        elements.

        """
        method_options, matplotlib_options = parse_kwargs(kwargs, TriMesh.plot_options)

        # Plot the cells
        ax = method_options['ax'] or plt.figure().add_subplot()
        ax.set_aspect('equal')
        ax.triplot(self.vertices[:, 0], self.vertices[:, 1], self.cells, *args,
                   **matplotlib_options)

        # Show element labels, if requested
        if method_options['cell_labels']:
            for label, elem in enumerate(self.cells):
                x, y = np.mean(self.vertices[elem], axis=0)
                ax.text(x, y, str(label))
        if method_options['vertex_labels']:
            for label, vertex in enumerate(self.vertices):
                x, y = vertex
                ax.text(x, y, str(label))

        # Plot the cell sets, vertex sets, and show their labels in the legend, if requested
        if method_options['cell_sets']:
            for name, cells in self.cell_sets.items():
                cells_in_set = self.cells[cells]
                triangles = ax.triplot(self.vertices[:, 0], self.vertices[:, 1], cells_in_set, *args,
                                       color=np.random.random((3, )), **matplotlib_options)[0]
                if method_options['cell_legends']:
                    n_cell_set = len(self.cell_sets)
                    triangles.set_label(name)
                    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=n_cell_set,
                              frameon=False)
        if method_options['vertex_sets']:
            for name, vertices in self.vertex_sets.items():
                markercolor = np.random.random((3, ))
                vertices_in_set = self.vertices[vertices]
                points = ax.plot(vertices_in_set[:, 0], vertices_in_set[:, 1], marker='.',
                                 markerfacecolor=markercolor, markeredgecolor=markercolor,
                                 markersize=10, linestyle='None')[0]
                if method_options['vertex_legends']:
                    n_vertex_set = len(self.vertex_sets)
                    points.set_label(name)
                    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=n_vertex_set,
                              frameon=False)

    def plot_field(self, component, *args, show_mesh=True, **kwargs):
        """Plots a field on the mesh.

        The aim of this method is to support basic post-processing for finite element visualization.
        Only the basic contour plot type is available. For vector or tensor fields, the
        components to be plotted must be chosen. For faster and more comprehensive plotting
        capabilities, turn to well-established scientific visualization software,
        such as `ParaView <https://www.paraview.org>`_ or `Mayavi
        <http://docs.enthought.com/mayavi/mayavi/>`_. Another limitation of the :meth:`plot_field`
        method is that field values are assumed to be associated to the vertices of the mesh,
        which restricts us to :math:`P1` Lagrange elements.

        Parameters
        ----------
        component : int
            Positive integer, the selected component of the field to be plotted. Components are
            indexed from 0.
        show_mesh : bool, optional
            If True, the underlying mesh is shown. The default is True.
        ax : matplotlib.axes.Axes, optional
            The `Axes` instance the plot resides in. The default is None, in which case a new
            `Axes` within a new figure is created.

        Other Parameters
        ----------------
        See them described in the :meth:`plot` method.

        Returns
        -------
        None

        See Also
        --------
        plot

        Examples
        --------
        The following example considers the same type of mesh as in the example shown for
        :meth:`plot`.

        >>> msh = TriMesh(*TriMesh.sample_mesh(1))

        We pretend that the field is an analytical function, evaluated at the vertices.

        >>> field = lambda x, y: 1 - (x + y**2) * np.sign(x)
        >>> field = field(msh.vertices[:, 0], msh.vertices[:, 1])

        We associate this field to the mesh and plot it with and without the mesh

        >>> msh.associate_field(field, 'analytical field')
        >>> _, (ax1, ax2) = plt.subplots(1, 2)
        >>> msh.plot_field(0, 'bo-', ax=ax1, linewidth=1, markerfacecolor='k')
        >>> msh.plot_field(0, ax=ax2, show_mesh=False)
        >>> plt.show()

        """
        if self.field['values'] is None:
            raise Exception('Field is not associated to the mesh.')
        field = self.field['values']
        n_component = np.shape(field)[1]
        if component not in range(n_component):
            raise ValueError('Component must be in [0, {0}].'.format(n_component-1))
        method_options, _ = parse_kwargs(kwargs, TriMesh.plot_options)
        ax = method_options['ax'] or plt.figure().add_subplot()
        ax.set_aspect('equal')
        ax.tricontourf(self.vertices[:, 0], self.vertices[:, 1], field[:, component], 1000)
        if show_mesh:
            self.plot(*args, **kwargs)

    def write_inp(self, filename):
        """Writes the mesh to an Abaqus .inp file.

        Parameters
        ----------
        filename : str
            Path to the mesh file.

        Returns
        -------
        None

        """
        # Node coordinates
        inp_file = '*NODE\n'
        for global_node_num, coordinate in enumerate(self.vertices, 1):
            # TODO: avoid reallocation by temporary strings as done below
            inp_file += str(global_node_num) + ', ' + str(list(coordinate))[1:-1] + '\n'

        # Cell-vertex connectivities
        element_type = 'CPS3'  # 3-node linear plane stress element; can be changed in Abaqus
        inp_file += '*ELEMENT, TYPE=' + element_type + '\n'
        for global_elem_num, elem_vertices in enumerate(self.cells, start=1):
            # TODO: avoid reallocation by temporary strings as done below
            inp_file += str(global_elem_num) + ', ' + str(list(elem_vertices+1))[1:-1] + '\n'

        # Element groups
        element_sets = ''  # Grow a new string to avoid extensive reallocation
        for group_name, elements in self.cell_sets.items():
            element_set = ''  # grow a new string to avoid extensive reallocation
            # Header
            element_set += '\n*ELSET, ELSET=' + group_name + '\n'
            # Elements belonging to this group
            for i, element in enumerate(elements+1, 1):
                element_set += str(element) + ', '
                if i % 16 == 0:  # Abaqus allows at most 16 entries per line
                    element_set += '\n'
            element_sets += element_set
        inp_file += element_sets

        # Node groups
        node_sets = ''  # Grow a new string to avoid extensive reallocation
        for group_name, nodes in self.vertex_sets.items():
            node_set = ''  # grow a new string to avoid extensive reallocation
            # Header
            node_set += '\n*NSET, NSET=' + group_name + '\n'
            # Nodes belonging to this group
            for i, node in enumerate(nodes + 1, 1):
                node_set += str(node) + ', '
                if i % 16 == 0:  # Abaqus allows at most 16 entries per line
                    node_set += '\n'
            node_sets += node_set
        inp_file += node_sets

        # Write to file
        with open(filename, 'w') as target:
            target.write(inp_file)

    @staticmethod
    def sample_mesh(sample, param=100):
        """Provides sample meshes.

        Parameters
        ----------
        sample : int
            Integer, giving the sample mesh to be considered. Possibilities:
        param :
            Parameters to the sample meshes. Possibilities:

        Returns
        -------
        nodes : ndarray
            2D numpy array with 2 columns, each row corresponding to a vertex, and the two columns
            giving the Cartesian coordinates of the vertices.
        cells : ndarray
            Cell-vertex connectivity in a 2D numpy array, in which each row corresponds to a
            cell and the columns are the vertices of the cells. It is assumed that all the
            cells have the same number of vertices.

        """
        if sample == 1:
            n_vertex = param
            a, b, c, d = (-1, 1, 1, 2)
            x_vertices = (b-a) * np.random.rand(n_vertex) + a
            y_vertices = (d-c) * np.random.rand(n_vertex) + c
            vertices = np.stack((x_vertices, y_vertices), axis=1)
            cells = Triangulation(x_vertices, y_vertices).triangles
        elif sample == 2:
            vertices = np.array([[]])
            cells = np.array([[]])
        else:
            raise ValueError('Sample not available.')
        return vertices, cells


def _polygon_area(x, y):
    """Computes the signed area of a non-self-intersecting, possibly concave, polygon.

    Directly taken from http://rosettacode.org/wiki/Shoelace_formula_for_polygonal_area#Python

    Parameters
    ----------
    x, y : list
        Coordinates of the consecutive vertices of the polygon.

    Returns
    -------
    float
        Area of the polygon.

    Warnings
    --------
    If numpy vectors are passed as inputs, the resulting area is incorrect! WHY?

    Notes
    -----
    The code is not optimized for speed and for numerical stability. Intended to be used to compute
    the area of finite element cells, in which case the numerical stability is not an issue
    (unless the cell is degenerate). As this function is called possibly as many times as the
    number of cells in the mesh, no input checking is performed.

    Examples
    --------
    >>> _polygon_area([0, 1, 1], [0, 0, 1])
    0.5

    """
    return 1/2*(sum(i*j for i, j in zip(x, y[1:]+y[:1])) -
                sum(i*j for i, j in zip(x[1:]+x[:1], y)))


class Polygon:
    """Represents a polygon.

    This class works as expected as long as the given polygon is `simple`, i.e. it is not
    self-intersecting and does not contain holes.

    A simple class that has `numpy` and `matplotlib` (for the visualization) as the only
    dependencies. It does not want to provide extensive functionalities (for those, check out
    `Shapely <https://github.com/Toblerity/Shapely>`_). The polygon is represented by its
    vertices, given in a consecutive order.

    Parameters
    ----------
    vertices : ndarray
        2D numpy array with 2 columns, each row corresponding to a vertex, and the two columns
        giving the Cartesian coordinates of the vertex.

    Raises
    ------
    Exception
        If all the vertices of the polygon lie along the same line.
        If the polygon is not given in R^2.
    ValueError
        If the polygon does not have at least 3 vertices.

    Examples
    --------
    Try to give a "polygon", in which all vertices are collinear

    >>> poly = Polygon(np.array([[0, 0], [1, 1], [2, 2]]))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Exception: All vertices are collinear. Not a valid polygon.

    Now we give a valid polygon:

    >>> pentagon = Polygon(np.array([[2, 1], [0, 0], [0.5, 3], [-1, 4], [3, 5]]))

    Use Python's `print` function to display basic information about a polygon:

    >>> print(pentagon)
    A non-convex polygon with 5 vertices, oriented clockwise.

    """
    # Default settings for plotting the polygon
    plot_options = {'ax': None,  # Axes object the mesh is plotted
                    'show_axes': True,  # show the coordinate axes
                    'vertex_labels': False  # show the vertex numbers
                    }

    def __init__(self, vertices):
        n_vertex, dimension = np.shape(vertices)
        if dimension != 2:
            raise Exception('Polygon vertices must lie on a plane')
        if n_vertex < 3:
            raise ValueError('A polygon consists of at least 3 vertices.')
        if is_collinear(vertices):
            raise Exception('All vertices are collinear. Not a valid polygon.')
        self.vertices = vertices
        self.n_vertex = n_vertex

    def area(self):
        """Signed area of the polygon.

        The signed area is computed by the shoelace formula [1]_

        .. math::
            A = \\frac{1}{2}\sum\limits_{i=1}^N (x_i y_{i+1} - x_{i+1} y_i)

        Returns
        -------
        float
            Signed area.

        References
        ----------
        .. [1] http://paulbourke.net/geometry/polygonmesh/centroid.pdf

        Examples
        --------
        >>> poly = Polygon(np.array([[0, 0], [1, 0], [1, 1], [-1, 1]]))
        >>> poly.area()
        1.5
        >>> poly = Polygon(np.array([[-1, 1], [1, 1], [1, 0], [0, 0]]))
        >>> poly.area()
        -1.5

        """
        vertex_idx = np.arange(-1, self.n_vertex)  # close the polygon
        x = self.vertices[vertex_idx, 0]
        y = self.vertices[vertex_idx, 1]
        return 1/2*sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(self.n_vertex))

    def centroid(self):
        """Centroid of the polygon.

        The centroid is computed according to the following formula [2]_

        .. math::
            C_x = \\frac{1}{6A}\sum\limits_{i=1}^N (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)

            C_y = \\frac{1}{6A}\sum\limits_{i=1}^N (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)

        where :math:`A` is the signed area determined by the :meth:`area` method and :math:`x_i,
        y_i` are the vertex coordinates with :math:`x_{N+1}=x_1` and :math:`y_{N+1}=y_1`.

        Returns
        -------
        tuple
            2-tuple, the coordinates of the centroid.

        References
        ----------
        .. [2] http://paulbourke.net/geometry/polygonmesh/centroid.pdf

        Examples
        --------
        >>> poly = Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        >>> poly.centroid()
        (0.5, 0.5)
        >>> poly = Polygon(np.array([[2, 1], [0, 0], [0.5, 3], [-1, 4], [3, 5]]))
        >>> poly.centroid()  # doctest: +ELLIPSIS
        (1.254..., 2.807...)

        """
        vertex_idx = np.arange(-1, self.n_vertex)  # close the polygon
        x = self.vertices[vertex_idx, 0]
        y = self.vertices[vertex_idx, 1]
        a = self.area()
        c_x = sum((x[i] + x[i+1]) * (x[i]*y[i+1] - x[i+1]*y[i]) for i in range(self.n_vertex))
        c_y = sum((y[i] + y[i+1]) * (x[i]*y[i+1] - x[i+1]*y[i]) for i in range(self.n_vertex))
        return 1/(6*a)*c_x, 1/(6*a)*c_y

    def diameter(self, definition='set'):
        """Diameter of the polygon.

        Multiple definitions are supported for the diameter:

        - `Diameter of a set`. The polygon is considered as a set :math:`A` of points comprised
        of the polygon vertices. Let :math:`(X,d)` be a metric space. The diameter of the set is
        defined as

        .. math::
            \mathrm{diam}(A) = \sup\{ d(x,y)\ |\ x, y \in A\}.
            :label: setDiameter

        Here, the Euclidean metric is used.

        - `Equivalent diameter`. Diameter of the circle of the same area as that of the polygon.

        Parameters
        ----------
        definition : {'set', 'equivalent'}, optional
            The default is 'set'.

        Returns
        -------
        float
            Diameter of the polygon, based on the chosen definition.

        Notes
        -----
        When :code:`definition` is 'set', computing the diameter by :math:numref:`setDiameter` is
        `equivalent <https://cs.stackexchange.com/questions/23646/why-are-the-two-farthest-points
        -vertices-of-the-convex-hull>`_ to determining the distance of the furthest points in the
        convex hull of :math:`A`. Therefore, the diameter will always be the distance between two
        points on the convex hull of :math:`A`. Then for each vertex of the hull finding which
        other hull vertex is farthest away from it, the `rotating caliper` algorithm can be used.
        Our brute-force method is simpler as it needs neither the convex hull nor the rotating
        caliper algorithm: all the pairwise distances among the polygon vertices are computed and
        the largest one is chosen. Pair of points a maximum distance apart. Since the polygons in
        our applications do not have that many vertices, this simplistic approach is a viable
        alternative.

        Examples
        --------
        >>> poly = Polygon(np.array([[2, 5], [0, 1], [4, 3], [4, 5]]))
        >>> poly.diameter('set')  # doctest: +ELLIPSIS
        5.6568542...
        >>> poly.diameter('equivalent')  # doctest: +ELLIPSIS
        3.1915382...
        >>> poly = Polygon(np.array([[2, 1], [3, -4], [-1, -1], [-4, -2], [-3, 0]]))
        >>> poly.diameter('set')  # doctest: +ELLIPSIS
        7.2801098...
        >>> poly.diameter('equivalent')  # doctest: +ELLIPSIS
        4.2967398...

        """
        if definition == 'set':
            return sqrt(np.max(distance_matrix(self.vertices)))
        elif definition == 'equivalent':
            return sqrt(4*abs(self.area())/pi)
        else:
            raise ValueError('Diameter definition unknown. Choose between "set" or "equivalent".')

    def orientation(self):
        """Orientation of the polygon.

        Returns
        -------
        str
            'cw' if the polygon has clockwise orientation, 'ccw' if counter-clockwise.

        """
        cross_product = np.cross(self.vertices[1] - self.vertices[0],
                                 self.vertices[2] - self.vertices[1])
        return 'ccw' if cross_product > 0 else 'cw'

    def is_convex(self):
        """Decides whether the polygon is convex.

        Returns
        -------
        bool
            True if the polygon is convex. False otherwise.

        Notes
        -----
        The algorithm works by checking if all pairs of consecutive edges in the polygon are
        either all clockwise or all counter-clockwise oriented. This method is valid only for
        simple polygons. The implementation follows `this code
        <https://github.com/crm416/point-location/blob/01b5c0f2105237e7108730d0e0db6213c0aadfbf/
        geo/shapes.py#L168>`_, extended for the case when two consecutive edges are collinear.
        If the polygon was not simple, a more complicated algorithm would be needed, see e.g.
        `here <https://stackoverflow.com/a/45372025/4892892>`_.

        Examples
        --------
        A triangle is always convex:

        >>> poly = Polygon(np.array([[1, 1], [0, 1], [0, 0]]))
        >>> poly.is_convex()
        True

        Let us define a concave deltoid:

        >>> poly = Polygon(np.array([[-1, -1], [0, 1], [1, -1], [0, 5]]))
        >>> poly.is_convex()
        False

        Give a polygon that has two collinear edges:

        >>> poly = Polygon(np.array([[0.5, 0], [1, 0], [1, 1], [0, 1], [0, 0]]))
        >>> poly.is_convex()
        True

        """
        target = None  # pairwise orientation of the edges, True if counter-clockwise
        target_set = False  # True if the truth value of `target` has been set
        for i in range(self.n_vertex):  # for every consecutive three vertices
            A = self.vertices[i]
            B = self.vertices[(i + 1) % self.n_vertex]
            C = self.vertices[(i + 2) % self.n_vertex]
            cross_product = np.cross(B - A, C - B)  # the orientation is implied from it
            if not isclose(cross_product, 0):  # otherwise the orientation is not accurate
                is_ccw = cross_product > 0
                if not target_set:
                    target = is_ccw
                    target_set = True
                else:
                    if is_ccw != target:  # not all pairs of edges have the same orientation
                        return False
        return True

    def plot(self, *args, **kwargs):
        """Plots the polygon.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The `Axes` instance the polygon resides in. The default is None, in which case a new
            `Axes` within a new figure is created.

        Other Parameters
        ----------------
        show_axes : bool, optional
            If True, the coordinate system is shown. The default is True.
        vertex_labels : bool, optional
            If True, vertex labels are shown. The default is False.
        args, kwargs : optional
            Additional arguments and keyword arguments to be specified. Those arguments are the
            ones supported by :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        None

        Examples
        --------
        Consider the pentagon used in the example of the constructor. Plot it in black with
        red diamond symbols representing its vertices. Moreover, display the vertex numbers
        and do not show the coordinate system.

        >>> pentagon = Polygon(np.array([[2, 1], [0, 0], [0.5, 3], [-1, 4], [3, 5]]))
        >>> pentagon.plot('k-d', vertex_labels=True, markerfacecolor='r', show_axes=False)
        >>> plt.show()

        """
        method_options, matplotlib_options = parse_kwargs(kwargs, Polygon.plot_options)
        ax = method_options['ax'] or plt.figure().add_subplot()
        ax.set_aspect('equal')
        vertex_idx = np.arange(-1, self.n_vertex)  # close the polygon
        ax.plot(self.vertices[vertex_idx, 0], self.vertices[vertex_idx, 1],
                *args, **matplotlib_options)
        if method_options['vertex_labels']:  # show the vertex numbers
            for label, vertex in enumerate(self.vertices, 1):
                x, y = vertex
                ax.text(x, y, str(label))
        if not method_options['show_axes']:  # do not show the axes
            ax.set_axis_off()

    def __str__(self):
        convexity = ' convex' if self.is_convex() else ' non-convex'
        orientation = ' clockwise' if self.orientation() == 'cw' else ' counter-clockwise'
        return 'A' + convexity + ' polygon with ' + str(self.n_vertex) + \
               ' vertices, oriented' + orientation + '.'


def is_collinear(points, tol=None):
    """Decides whether a set of points is collinear.

    Works in any dimensions.

    Parameters
    ----------
    points : ndarray
        2D numpy array with N columns, each row corresponding to a point, and the N columns
        giving the Cartesian coordinates of the point.

    tol : float, optional
        Tolerance value passed to numpy's `matrix_rank` function. This tolerance gives the
        threshold below which SVD values are considered zero.

    Returns
    -------
    bool
        True for collinear points.

    See Also
    --------
    :func:`numpy.linalg.matrix_rank`

    Notes
    -----
    The algorithm for three points is from `Tim Davis <https://nl.mathworks.com/matlabcentral/
    discussions/b-loren/127448-loren-on-the-art-of-matlab-collinearity/21239#reply_21239>`_.

    Examples
    --------
    Two points are always collinear

    >>> is_collinear(np.array([[1, 0], [1, 5]]))
    True

    Three points in 3D which are supposed to be collinear (returns false due to numerical error)

    >>> is_collinear(np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]]), tol=0)
    False

    The previous example with looser tolerance

    >>> is_collinear(np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]]), tol=1e-14)
    True

    """
    return np.linalg.matrix_rank(points - points[0, :], tol=tol) < 2


def squared_distance(x, y):
    """Squared Euclidean distance between two points.

    For points :math:`x(x_1, ..., x_n)` and :math:`y(y_1, ... y_n)` the following metric is computed

    .. math::
        \sum\limits_{i=1}^n (x_i - y_i)^2

    Parameters
    ----------
    x, y : ndarray
        1D numpy array, containing the coordinates of the two points.

    Returns
    -------
    float
        Squared Euclidean distance.

    See Also
    --------
    distance_matrix

    Examples
    --------
    >>> squared_distance(np.array([0, 0, 0]), np.array([1, 1, 1]))
    3.0

    """
    return float(((x - y) ** 2).sum())


def distance_matrix(points):
    """A symmetric square matrix, containing the pairwise squared Euclidean distances among points.

    Parameters
    ----------
    points : ndarray
        2D numpy array with 2 columns, each row corresponding to a point, and the two columns
        giving the Cartesian coordinates of the points.

    Returns
    -------
    dm : ndarray
        Distance matrix.

    See Also
    --------
    squared_distance

    Examples
    --------
    >>> points = np.array([[1, 1], [3, 0], [-1, -1]])
    >>> distance_matrix(points)
    array([[ 0.,  5.,  8.],
           [ 5.,  0., 17.],
           [ 8., 17.,  0.]])

    """
    dm = np.asarray([[squared_distance(p1, p2) for p2 in points] for p1 in points])
    np.fill_diagonal(dm, 0.0)
    return dm
