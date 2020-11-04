# -*- coding: utf-8 -*-
"""
Extracting and processing meshes from .med files.
The functions were tested on the MEDCoupling API, version 9.4.0.

.. todo:: Support renumbering (https://docs.salome-platform.org/latest/dev/MEDCoupling/user/html/data_optimization.html).

Getting help:

* This module relies on the Python interface of MEDCoupling. `Click here <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/index.html>`_ for the latest documentation.
* `User's manual <https://docs.salome-platform.org/latest/dev/MEDCoupling/user/html/index.html>`_ for the Python interface
* To know more about the MED file format, which is a specialization of HDF5, see the `documentation <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html>`_.
  For a discussion on the relation between the MED format and the APIS, see `this page
  <https://www.salome-platform.org/user-section/about/med>`_ and `that one
  <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/library.html>`_.
* The definitions, such as `group`, used in this module are from the `development guide <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/glossary.html>`_.
* A (mostly English) `tutorial <https://docs.salome-platform.org/latest/dev/MEDCoupling/tutorial/index.html>`_ for the Python interface to MEDCoupling is also useful.
  Particularly interesting are the `mesh manipulation examples <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/medcouplingpyexamples.html#ExamplesMeshes>`_
* `Main page <https://salome-platform.org/user-section/documentation/current-release>`_ of the documentation

Functions
---------
.. autosummary::
    :nosignatures:
    :toctree: functions/

    read_mesh
    get_nodes
    get_elements

"""

import warnings

import numpy as np

try:
    from MEDLoader import MEDFileData
    # from MEDCoupling import MEDCouplingUMesh, MEDCouplingMesh, MEDCouplingUMeshCellIterator
except:
    raise ImportError('This module requires the MEDCoupling module from Salome.')


def read_mesh(filename):
    """Reads a mesh file in .med format.
    Only one mesh, the first one, is supported. However, that mesh can contain groups.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    MEDFileUMesh
        Represents an unstructured mesh. For details, see the manual on
        https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/classMEDCoupling_1_1MEDFileUMesh.html
    """
    # Read the mesh
    data = MEDFileData(filename)
    mesh = data.getMeshes()
    mesh_names = mesh.getMeshesNames()
    n_mesh = len(mesh_names)
    if n_mesh > 1:
        warnings.warn('{0} meshes detected, only the fist one is considered.'.format(n_mesh),
                      RuntimeWarning)
    mesh = mesh.getMeshWithName(mesh_names[0])
    if mesh.getMeshDimension() != 2:
        warnings.warn('Only 2D meshes are supported.', RuntimeWarning)
    return mesh


def get_nodes(mesh):
    """Obtains the nodes and the node groups of a mesh.

    Parameters
    ----------
    mesh : MEDFileUMesh
        Unstructured mesh.

    Returns
    -------
    nodes : ndarray
        2D numpy array with 2 columns, each row corresponding to a node, and the two columns
        giving the Cartesian coordinates of the nodes.
    node_groups : dict
        The keys in the dictionary are the node group names, while the values are list of integers,
        giving the nodes that belong to the particular group.

    See Also
    --------
    :func:`get_elements`,
    `getGroupArr <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/classMEDCoupling_1_1MEDFileMesh.html#a4398c05f015e52d0d380eb39c6e4b942>`_

    """
    # Get the coordinates of all the nodes in the mesh
    nodes = mesh.getCoords().toNumPyArray()
    # Get the node groups
    group_names = mesh.getGroupsOnSpecifiedLev(1)
    node_groups = {
        group_name: mesh.getGroupArr(1, group_name).toNumPyArray()
        for group_name in group_names
    }

    return nodes, node_groups


def get_elements(mesh, numbering='global'):
    """Obtains the elements for each group of a mesh.

    Elements of the same dimension as the mesh are collected (e.g. faces for a 2D mesh).

    .. todo:: put those elements that do not belong to any group into an automatically created group
    .. todo:: support ordering `elements` in alphabetical order
    .. todo:: implement the `'global'` strategy

    Parameters
    ----------
    mesh : MEDFileUMesh
        Unstructured mesh.
    numbering : {'global'}, optional
        Determines how to allocate element numbers in the mesh.
            'global': numbers the elements without taking into account which group they belong to.
            Use this strategy if you are not sure whether an element belongs to more than one group.
            'group': numbers the elements group-wise. This is much faster than the `'global'`
            strategy, but use this option if you are sure that the groups of the mesh do not
            contain common elements.
        The default is 'global'.

    Returns
    -------
    elements : ndarray
        Element-node connectivities in a 2D numpy array, in which each row corresponds to an
        element and the columns are the nodes of the elements. It is assumed that all the
        elements have the same number of nodes.
    element_groups : dict
        The keys in the dictionary are the element group names, while the values are list of
        integers, giving the elements that belong to the particular group.

    Warnings
    --------
    Currently, elements that do not fit into any groups are discarded.

    See Also
    --------
    get_nodes
    change_node_numbering

    Notes
    -----
    The element-node connectivities are read from the mesh. If you want to change the ordering
    of the nodes, use the :func:`change_node_numbering` function.

    Both this and the :func:`get_nodes` function relies on `getGroupsOnSpecifiedLev
    <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer
    /classMEDCoupling_1_1MEDFileMesh.html#a2d59097b6d14b95c7d2aeee9f39b0438>`_ to obtain the groups
    based on a parameter, called `meshDimRelToMaxExt`. This parameter designates the relative
    dimension of the mesh entities whose IDs are required. If it is 1, it denotes the nodes. If
    0, entities of the same dimension as the mesh are meant (e.g. group of volumes for a 3D mesh,
    or group of faces for a 2D mesh). When -1, entities of spatial dimension immediately below
    that of the mesh are collected (e.g. group of faces for a 3D mesh, or group of edges for a
    2D mesh). For -2, entities of two dimensions below that of the mesh are fetched (e.g. group of
    edges for a 3D mesh).

    """
    group_names = mesh.getGroupsOnSpecifiedLev(0)
    n_element = mesh.getDistributionOfTypes(0)[1]
    elements = np.empty((n_element, 3), dtype=int)
    element_groups = {}
    if numbering == 'group':  # process the elements group-wise
        i = 0  # indexing elements in the whole mesh
        for group_name in group_names:
            local_mesh = mesh.getGroup(0, group_name)
            n_element_in_group = local_mesh.getNumberOfCells()
            local_elements = np.empty(n_element_in_group, dtype=int)
            for j, cell in enumerate(local_mesh):  # j: indexing elements within a group
                elements[i, :] = list(cell.getAllConn()[1:])
                local_elements[j] = i
                i += 1
            element_groups[group_name] = local_elements
        if i < n_element:  # some elements are not part of any group
            elements = elements[0:i]
            warnings.warn('Elements that do not belong to any of the groups were discarded.')
    elif numbering == 'global':
        raise ValueError('Strategy not yet implemented.')
    else:
        raise ValueError('Unknown strategy. Choose either "global" or "group".')
    return elements, element_groups
