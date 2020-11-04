#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes
-------
.. autosummary::
    :nosignatures:
    :toctree: classes/

    SkeletonGeometry
    QuadSkeletonGeometry
    TriSkeletonGeometry
    FixedDict
    OOF2

"""
import os.path
import re
from collections import namedtuple
from abc import ABC, abstractmethod

import numpy as np


class SkeletonGeometry(ABC):

    @abstractmethod
    def __init__(self, leftright_periodicity, topbottom_periodicity):
        self.leftright_periodicity = leftright_periodicity
        self.topbottom_periodicity = topbottom_periodicity


class QuadSkeletonGeometry(SkeletonGeometry):

    def __init__(self, leftright_periodicity=False,
                 topbottom_periodicity=False):
        super().__init__(leftright_periodicity, topbottom_periodicity)

    def __repr__(self):
        return "QuadSkeleton(left_right_periodicity={1}, " \
            "top_bottom_periodicity={2})".format(self.leftright_periodicity,
                                                 self.topbottom_periodicity)

class TriSkeletonGeometry(SkeletonGeometry):

    def __init__(self, leftright_periodicity=False,
                 topbottom_periodicity=False,
                 arrangement="conservative"):
        super().__init__(leftright_periodicity, topbottom_periodicity)
        self.arrangement = arrangement

    def __repr__(self):
        return "TriSkeleton(arrangement='{0}', left_right_periodicity={1}, " \
            "top_bottom_periodicity={2})".format(self.arrangement,
                                                 self.leftright_periodicity,
                                                 self.topbottom_periodicity)


class FixedDict:
    "https://stackoverflow.com/a/14816446/4892892"

    def __init__(self, dictionary):
        self._dictionary = dictionary

    def __setitem__(self, key, item):
        if key not in self._dictionary:
            raise KeyError("The key {} is not defined.".format(key))
        self._dictionary[key] = item

    def __getitem__(self, key):
        return self._dictionary[key]

    def __repr__(self):
        return self._dictionary.__repr__()

    def __str__(self):
        return self._dictionary.__str__()


nt = namedtuple("modules", ["image", "microstructure", "material",
                            "pixelgroup", "skeleton"])
image = FixedDict({"name": None, "extension": None, "fullname": None})
microstructure = FixedDict({"name": None, "pixelgroups": None})
material = []
# material = FixedDict({"names": [], "pixelgroups": []})
pixelgroup = {}
skeleton = FixedDict({"name": None, "geometry": None})

class OOF2:
    script = []
    _modules = nt._make([image, microstructure, material, pixelgroup, skeleton])
    _state = FixedDict({"image_read": False, "microstructure_created": False,
                        "groups_created": False, "material_created": False,
                        "pixelgroups_created": False, "skeleton_created": False,
                        "script_saved": False})

    def read_image(self, label_image):
        # TODO: check somehow if the input is a label image
        filename, extension = os.path.splitext(os.path.basename(label_image))
        self._modules.image["name"] = filename
        self._modules.image["extension"] = extension
        self._modules.image["fullname"] = filename + extension
        self._state["image_read"] = True

    def create_microstructure(self, name=None):
        """Creates a microstructure from an image.

        Parameters
        ----------
        name : str, optional
            Path to the image on which the microstucture is created,
            file extension included. If not given, the microstructure is given
            the same name as the input image.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not self._state["image_read"]:
            raise Exception("Load an image first with the `read_image` method.")
        if name:
            if not isinstance(name, str):
                raise Exception("File name must be a string")
        else:
            name = self._modules.image["name"]
        self._modules.microstructure["name"] = name
        self.script.append("OOF.Microstructure.Create_From_ImageFile"
                           "(filename='{0}', microstructure_name='{1}', "
                           "height=automatic, width=automatic)"
                           .format(self._modules.image["fullname"],
                                   self._modules.microstructure["name"]))
        self._state["microstructure_created"] = True

    def pixel2group(self):
        self.script.append("OOF.Image.AutoGroup(image='{0}', name_template='%n')"
                           .format(self._modules.microstructure["name"] + ":" +
                                   self._modules.image["fullname"]))
        self._state["groups_created"] = True

    def save_microstructure(self, name=None):
        if not self._state["microstructure_created"]:
            raise Exception("Create the mictrostructure first with the "
                            "`create_microstructure` method.")
        if name:
            if not isinstance(name, str):
                raise Exception("File name must be a string")
        else:
            name = self._modules.microstructure["name"] + '.txt'
        self.script.append("OOF.File.Save.Microstructure(filename='{0}', "
                           "mode='w', format='script', microstructure='{1}')"
                           .format(name, self._modules.microstructure["name"]))

    def load_pixelgroups(self, microstructure_file):
        """

        Parameters
        ----------
        microstructure_file : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self._state["pixelgroups_created"]:
            raise Exception("Pixelgroups already created.")
            # TODO: other option is to overwrite the existing pixel group.
            # However, in this case, all subsequent steps (e.g. materials2groups)
            # need to be cancelled.
        # Identify which pixel belongs to which category
        with open(microstructure_file, 'r') as microstructure:
            data = microstructure.readlines()
            category = []; group = []
            for line in data:
                if "OOF.LoadData.Microstructure.DefineCategory.PixelGroups" in line:
                    m = re.match(".+category=([\w]+),\sgroups=\['([\w]+)'\]", line)
                    category.append(int(m.group(1)))
                    group.append(int(m.group(2)))
                elif "OOF.LoadData.Microstructure.Categories" in line:
                    m = re.match(r".+categories=(.+)\)", line)
                    nestedlist = eval(m.group(1))
                    pixelcategory = np.array(nestedlist)
        # Obtain the pixel-group classification using the category-group relation
        pixelgroup = pixelcategory.copy()
        for i in range(len(category)):
            pixelgroup[pixelcategory == category[i]] = group[i]
        self._modules.microstructure["pixelgroups"] = pixelgroup
        self._modules.pixelgroup.update(dict.fromkeys(group))
        self._state["pixelgroups_created"] = True

    def save_pixelgroups(self, name=None):
        """

        Parameters
        ----------
        name : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if name:
            if not isinstance(name, str):
                raise Exception("File name must be a string")
        else:
            name = self._modules.image["name"]
        np.save(name, self._modules.microstructure["pixelgroups"])

    def create_material(self, name):
        """

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if name in self._modules.material:
            raise Exception("Material `{0}` already exists. Give another name."
                            .format(name))
        self._modules.material.append(name)
        self.script.append("OOF.Material.New(name='{0}', material_type='bulk')"
                           .format(name))
        self._state["material_created"] = True

    def materials2groups(self, materials, groups=None):
        """

        Parameters
        ----------
        materials : list of str
            DESCRIPTION.
        groups : list of int, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if groups:
            if len(materials) == len(groups):
                for mat, gr in zip(materials, groups):
                    if mat in self._modules.material:
                        self.script.append("OOF.Material.Assign(material='{0}', "
                                           "microstructure='{1}', pixels='{2}')"
                                           .format(mat, self._modules.microstructure["name"], gr))
                        self._modules.pixelgroup[gr] = mat
                        # self._modules.material[mat] = gr
                    else:
                        raise Exception("Material `{0}` does not exist.".format(mat))
            else:
                raise Exception("The number of materials and the number of groups must agree.")
        else:
            n_material = len(materials)
            if n_material == 1:
                mat = materials[0]
                if mat in self._modules.material:
                    # Associate the material to all the pixel groups
                    for gr in self._modules.pixelgroup:
                        self.script.append("OOF.Material.Assign(material='{0}', "
                                           "microstructure='{1}', pixels='{2}')"
                                           .format(mat, self._modules.microstructure["name"], gr))
                        self._modules.pixelgroup[gr] = mat
                else:
                    raise Exception("Material `{0}` does not exists.".format(mat))
            else:  # must give groups to avoid ambiguity
                raise Exception("{0} number of groups expected.".format(n_material))

    def create_skeleton(self, nelem_x, nelem_y, geometry, name=None):
        if not self._state["microstructure_created"]:
            raise Exception("Create the mictrostructure first with the "
                            "`create_microstructure` method.")
        if name:
            if not isinstance(name, str):
                raise Exception("File name must be a string")
        else:
            name = self._modules.image["name"]
        self._modules.skeleton["name"] = name
        self.script.append("OOF.Skeleton.New(name='{0}', microstructure='{1}', "
                           "x_elements={2}, y_elements={3}, "
                           "skeleton_geometry={4})"
                           .format(self._modules.skeleton["name"],
                                   self._modules.microstructure["name"],
                                   nelem_x, nelem_y, repr(geometry)))
        self._state["skeleton_created"] = True

    def write_script(self, name=None):
        if name:
            if not isinstance(name, str):
                raise Exception("File name must be a string")
        else:
            name = self._modules.image["name"] + ".script"
        with open(name, 'w') as script:
            script.writelines("\n".join(self.script))
        self._state["script_saved"] = True

    def show(self):
        """

        Returns
        -------
        None.

        """
        print("\n".join(self.script))

    def __str__(self):
        """Customizes how the object is printed.
        Displays basic information about the materials.
        For detailed information, use the `show` method.

        Returns
        -------
        str
            DESCRIPTION.

        """
        # display = ''.join(display)
        # return display
        return ""
