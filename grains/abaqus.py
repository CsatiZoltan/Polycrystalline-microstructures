"""
This module contains the Material class, responsible for managing materials
and interfacing with Abaqus.
"""

# TODO:
#     - finish the `create` method
#     - put the methods in order of importance
#     - update the docstrings
#     - finish the class docstring
#     - add to version control
#     - consider using the extract function
#     - allow Abaqus keywords taking options (see the previous TODO)

import os
import re


class Material:
    """Add materials to an Abaqus input file.
    Requirements: be able to
    - create an empty .inp file, containing only the materials
    - add materials to an existing .inp file

    TODO: only document public attributes!
    Attributes
    ----------
    materials : dict
        materials, their behaviors and their parameters are stored here.
        Intended for internal representation. To view them, use the `show`
        method, to write them to an input file, use the `write` method.

    Methods
    -------
    read(inp_file)
        Reads material data from an Abaqus .inp file.
    write(output_file=None)
        Writes material data to an Abaqus .inp file.
    remove(inp_file, output_file=None)
        Removes material definitions from an Abaqus .inp file.
    create(inp_file)
        Creates empty Abaqus .inp file.
    show()
        Shows material data as it appears in the Abaqus .inp file.

    Notes
    -----
    The aim of this class is to programmatically add, remove, modify materials
    in a format understandable by Abaqus. This class does not target editing
    materials through a GUI (that can be done in Abaqus CAE).

    """
    def __init__(self, from_Abaqus=False):
        """

        Parameters
        ----------
        from_Abaqus : bool, optional
            True if the input file was generated by Abaqus. The default is False.
            Abaqus generates input files with a consistent format. This allows
            certain optimizations: the input file may not need to be completely
            traversed to extract the materials. Third-party programs sometimes
            generate .inp files, which are valid but do not follow the Abaqus
            pattern. In this case, it cannot be predicted where the material
            definition ends in the file -- the whole file needs to be traversed.

        Returns
        -------
        None.

        """
        self.inp_file = ''
        self.state = {'begin': None, 'end': None, 'read': False}
        self.materials = {}
        self.is_greedy = from_Abaqus

    def add_material(self, material):
        """Defines a new material by its name.

        Parameters
        ----------
        material : str
            Name of the material to be added. A material can have multiple
            behaviors (e.g. elastoplastic).

        Returns
        -------
        None.
        """
        if material in self.materials:
            print('Material {0} already exists.'.format(material))
        else:
            self.materials[material] = {}

    def add_linearelastic(self, material, E, nu):
        """Adds linear elastic behavior to a given material.

        Parameters
        ----------
        material : str
            Name of the material the behavior belongs to.
        E : int, float
            Young's modulus.
        nu : int, float
            Poisson's ratio.

        Returns
        -------
        None.

        """
        if material in self.materials:
            if 'Elastic' in self.materials[material]:
                print('Elastic behavior already exists for material {0}. '
                      'Remove it first to add the new one.'.format(material))
            else:
                self.materials[material]['Elastic'] = [[str(E), ' ' + str(nu)]]
        else:
            print('Material {0} does not exist. Behavior not added.'.format(material))

    def add_plastic(self, material, sigma_y, epsilon_p):
        """Adds metal plasticity behavior to a given material.

        Parameters
        ----------
        material : str
            Name of the material the behavior belongs to.
        sigma_y : int, float
            Yield stress.
        epsilon_p : int, float
            Plastic strain.

        Returns
        -------
        None.

        """
        if material in self.materials:
            if 'Plastic' in self.materials[material]:
                print('Plastic behavior already exists for material {0}. '
                      'Remove it first to add the new one.'.format(material))
            else:
                self.materials[material]['Plastic'] = [[str(sigma_y), ' ' + str(epsilon_p)]]
        else:
            print('Material {0} does not exist. Behavior not added.'.format(material))

    def read(self, inp_file):
        """Reads material data from an Abaqus .inp file.

        Parameters
        ----------
        inp_file : str
            Abaqus input (.inp) file to be created.

        Returns
        -------
        None.

        Notes
        -----
        - This method is designed to read material data. Although the logic
          could be used to process other properties (parts, assemblies, etc.)
          in an input file, they are not yet implemented in this class.
        - This method assumes that the input file is valid. If it is, the
          material data can be extacted. If not, the behavior is undefined:
          the program can crash or return garbage. This is by design:
          the single responsibility principle dictates that the validity of the
          input file must be provided by other methods. If the input file was
          generated from within Abaqus CAE, it is guaranteed to be valid.
          The `write` method of this class also ensures that the resulting
          input file is valid. This design choice also makes the program logic
          simpler.
          For valid syntax in the input file, check the Input Syntax Rules
          section in the Abaqus user's guide.
        - To read material data from an input file, one has to identify the
          structure of .inp files in Abaqus. Abaqus is driven by keywords and
          corresponding data. For a list of accepted keywords, consult the
          Abaqus Keywords Reference Guide.
          There are three types of input lines in Abaqus:
              - keyword line: begins with a star, followed by the name of the
                keyword. Parameters, if any, are separated by commas and are
                given as parameter-value pairs. Keywords and parameters are not
                case sensitive. Example:
                    *ELASTIC, TYPE=ISOTROPIC, DEPENDENCIES=1
                Some keywords can only be defined once another keyword has
                already been defined. E.g. the keyword ELASTIC must come after
                MATERIAL in a valid .inp file.
              - data line: immediately follows a keyword line. All data items
                must be separated by commas. Example:
                    -12.345, 0.01, 5.2E-2, -1.2345E1
              - comment line: starts with ** and is ignored by Abaqus. Example:
                    ** This is a comment line
        - Internally, the materials are stored in a dictionary. It holds the
          material data read from the file. The keys in this dictionary are the
          names of the materials, and the values are dictionaries themselves.
          Each such dictionary stores a behavior for the given material.
          E.g. an elastoplastic material is governed by an elastic and a
          plastic behavior. The parameters for each behavior are stored in a list.

        """
        if self.state['read']:
            raise Exception('Material database already exists. Instantiate a '
                            'new Material object for another database.')
        validate_file(inp_file, 'read')
        in_material = False  ## true when reached the material definition block
        materials = {}  ## holds the materials read from the input file
        generalkeyword = re.compile('^\*\s?\w')  # any Abaqus keyword
        material = re.compile('\*material, name=([\w-]+)', re.IGNORECASE)
        behavior = re.compile('\*(plastic|elastic)', re.IGNORECASE)
        comment = re.compile('^[\s]?\*\*')
        is_greedy = self.is_greedy
        with open(inp_file, 'r') as file:
            for line_number, line in enumerate(file, 0):  # read the file line by line
                is_generalkeyword = generalkeyword.match(line)
                is_material = material.match(line)
                is_behavior = behavior.match(line)
                is_comment = comment.match(line)
                is_parameter = not is_generalkeyword and not is_comment
                is_material_param = is_parameter and in_material
                if is_material:
                    if not in_material:  # we entered the first material
                        in_material = True
                        begin = line_number
                    material_name = is_material.group(1)
                    materials[material_name] = {}
                elif is_behavior:
                    behavior_name = is_behavior.group(1)
                    materials[material_name][behavior_name] = []
                elif is_material_param:
                    params = line[0:-1].split(',')
                    materials[material_name][behavior_name].append(params)
                elif (is_generalkeyword or is_comment) and in_material and is_greedy:
                    # We were previously in a material definition section but
                    # now we detected a new keyword. In an Abaqus-generated
                    # .inp file this indicates that the material section has
                    # ended: no need to search for more
                    end = line_number-1
                    break
        if 'end' not in locals():  # input file ends with materials
            end = line_number
        self.inp_file = inp_file
        self.materials = materials
        self.state['begin'] = begin
        self.state['end'] = end
        self.state['read'] = True

    def write(self, output_file=None):
        """Writes material data to an Abaqus .inp file.

        Parameters
        ----------
        output_file : str, optional
            Output file name to write the modifications into.
            If not given, the original file name is appended with '_mod'.

        Returns
        -------
        None.

        Notes
        -----
        - If the output file name is the same as the input file, the original
          .inp file will be overwritten. This is strongly not recommended.
        - The whole content of the original input file is read to memory. It
          might be a problem for very large .inp files. In that case, a possible
          implementation could be the following:
              1. Remove old material data
              2. Append new material data to the proper position in the file
          Appending is automatically done at the end of the file. Moving the
          material description to the end of the file is not possible in general
          because defining materials cannot be done from any module, i.e. the
          *MATERIAL keyword cannot follow an arbitrary keyword. In this case,
          Abaqus throws an AbaqusException with the following message:
              It can be suboption for the following keyword(s)/level(s): model

        """
        if not self.materials:
            raise Exception('Material does not exist. Nothing to write.')
        if self.state["read"]:  # database was created by reading a .inp file
            with open(self.inp_file, 'r') as source:
                old = source.readlines()
            before = old[0:self.state['begin']]
            after = old[self.state['end']+1:]
            new = before + self.__format() + after
            if not output_file:
                name, extension = os.path.splitext(self.inp_file)
                output_file = name + '_mod' + extension
            validate_file(output_file, 'write')
            with open(output_file, 'w') as target:
                target.write(''.join(new))
        else:  # database was created manually
            if not output_file:
                dirname = os.path.split(os.path.abspath(__file__))[0]
                files = os.listdir()
                filename = 'materials.inp'
                i = 0
                while filename in files:
                    i += 1
                    filename = 'materials-{0}.inp'.format(i)
                output_file = filename
            validate_file(output_file, 'write')
            with open(output_file, 'w') as target:
                target.write(''.join(self.__format()))
        print('Material database saved as `{0}`.'.format(output_file))

    def create(self, inp_file):
        """Creates empty Abaqus .inp file.

        Parameters
        ----------
        inp_file : str
            Abaqus input file to be created. If an extension is not given, the
            default .inp is used.

        Returns
        -------
        None.

        """
        self.state['created'] = True

    @staticmethod
    def remove(inp_file, output_file=None):
        """Removes material definitions from an Abaqus .inp file.

        Parameters
        ----------
        inp_file : str
            Abaqus .inp file from which the materials should be removed.
        output_file : str, optional
            Output file name to write the modifications into.
            If not given, the original file name is appended with '_mod'.

        Returns
        -------
        None.

        """
        with open(inp_file, 'r') as source:
            a = Material()
            a.read(inp_file)
            old = source.readlines()
        before = old[0:a.state['begin']]
        after = old[a.state['end']+1:]
        new = before + after
        if not output_file:
            name, extension = os.path.splitext(inp_file)
            output_file = name + '_mod' + extension
        validate_file(output_file, 'write')
        with open(output_file, 'w') as target:
            target.write(''.join(new))

    def show(self):
        """Shows material data as it appears in the Abaqus .inp file.

        Returns
        -------
        None.

        """
        print(''.join(self.__format()))

    def __format(self):
        """Formats the material data in the Abaqus .inp format.
        The internal representation of the material data in converted to a
        string understood by Abaqus.

        Returns
        -------
        abaqus_format : list
            List of strings, each element of the list corresponding to a line
            (with \n line ending) in the Abaqus .inp file. In case of no
            material, an empty list is returned.

        Notes
        -----
        The output is a list so that further concatenation operations are easy.
        If you want a string, merge the elements of the list:
            output = ''.join(output)
        This is what the `show` method does.

        """
        abaqus_format = []
        for material_name, behaviors in self.materials.items():
            abaqus_format.append('*Material, name={0}\n'.format(material_name))
            for behavior_name, parameters in behaviors.items():
                abaqus_format.append('*{0}\n'.format(behavior_name))
                for parameter in parameters:
                    abaqus_format.append(','.join(parameter) + '\n')
        return abaqus_format

    @staticmethod
    def __isnumeric(x):
        """Decides if the input is a scalar number.

        Parameters
        ----------
        x : any type
            Input to be tested.

        Returns
        -------
        bool
            True if the given object is a scalar number.

        """
        return isinstance(x, (int, float, complex)) and not isinstance(x, bool)

    def __str__(self):
        """Customizes how the object is printed.
        Displays basic information about the materials.
        For detailed information, use the `show` method.

        Returns
        -------
        str
            DESCRIPTION.

        """
        n_material = len(self.materials)
        if n_material in {0, 1}:
            display = ['{0} material.\n'.format(n_material)]
        else:
            display = ['{0} materials.\n'.format(n_material)]
        for material_name in self.materials:
            display.append('    {0}\n'.format(material_name))
        display = ''.join(display)
        return display


class Geometry:
    """Modifies existing geometry.
    """

    def __init__(self):
        self.inp_file = ''
        self.state = {'begin': None, 'end': None, 'read': False}
        self.nodes = {}
    
    def read(self, inp_file):
        """Reads material data from an Abaqus .inp file.

        Parameters
        ----------
        inp_file : str
            Abaqus input (.inp) file to be created.

        Returns
        -------
        None.

        Notes
        -----
        - This method is designed to read material data. Although the logic
          could be used to process other properties (parts, assemblies, etc.)
          in an input file, they are not yet implemented in this class.
        - This method assumes that the input file is valid. If it is, the
          material data can be extacted. If not, the behavior is undefined:
          the program can crash or return garbage. This is by design:
          the single responsibility principle dictates that the validity of the
          input file must be provided by other methods. If the input file was
          generated from within Abaqus CAE, it is guaranteed to be valid.
          The `write` method of this class also ensures that the resulting
          input file is valid. This design choice also makes the program logic
          simpler.
          For valid syntax in the input file, check the Input Syntax Rules
          section in the Abaqus user's guide.
        - To read material data from an input file, one has to identify the
          structure of .inp files in Abaqus. Abaqus is driven by keywords and
          corresponding data. For a list of accepted keywords, consult the
          Abaqus Keywords Reference Guide.
          There are three types of input lines in Abaqus:
              - keyword line: begins with a star, followed by the name of the
                keyword. Parameters, if any, are separated by commas and are
                given as parameter-value pairs. Keywords and parameters are not
                case sensitive. Example:
                    *ELASTIC, TYPE=ISOTROPIC, DEPENDENCIES=1
                Some keywords can only be defined once another keyword has
                already been defined. E.g. the keyword ELASTIC must come after
                MATERIAL in a valid .inp file.
              - data line: immediately follows a keyword line. All data items
                must be separated by commas. Example:
                    -12.345, 0.01, 5.2E-2, -1.2345E1
              - comment line: starts with ** and is ignored by Abaqus. Example:
                    ** This is a comment line
        - Internally, the materials are stored in a dictionary. It holds the
          material data read from the file. The keys in this dictionary are the
          names of the materials, and the values are dictionaries themselves.
          Each such dictionary stores a behavior for the given material.
          E.g. an elastoplastic material is governed by an elastic and a
          plastic behavior. The parameters for each behavior are stored in a list.

        """
        if self.state['read']:
            raise Exception('Material database already exists. Instantiate a '
                            'new Material object for another database.')
        validate_file(inp_file, 'read')
        in_node = False  ## true when reached the node definition block
        nodes = []  ## holds the materials read from the input file
        generalkeyword = re.compile('^\*\s?\w')  # any Abaqus keyword
        node = re.compile('\*node', re.IGNORECASE)
        comment = re.compile('^[\s]?\*\*')
        with open(inp_file, 'r') as file:
            for line_number, line in enumerate(file, 0):  # read the file line by line
                is_generalkeyword = generalkeyword.match(line)
                is_node = node.match(line)
                is_comment = comment.match(line)
                is_parameter = not is_generalkeyword and not is_comment
                is_node_param = is_parameter and in_node
                if is_node:  # we entered the NODE definition block
                    in_node = True
                    begin = line_number
                elif is_node_param:
                    params = line[0:-1].split(',')
                    nodes.append(params)
                elif (is_generalkeyword or is_comment) and in_node:
                    # We were previously in a material definition section but
                    # now we detected a new keyword. In an Abaqus-generated
                    # .inp file this indicates that the material section has
                    # ended: no need to search for more
                    end = line_number-1
                    break
        if 'end' not in locals():  # input file ends with nodes
            end = line_number
        self.inp_file = inp_file
        self.nodes = nodes
        self.state['begin'] = begin
        self.state['end'] = end
        self.state['read'] = True

    def scale(self, factor):
        """Scales the geometry by modifying the coordinates of the nodes.

        Parameters
        ----------
        factor : float
            Each nodal coordinate is multiplied by this non-negative number.

        Returns
        -------
        None.

        Notes
        -----
        The modification happens in-place.

        """
        if not self.nodes:
            raise Exception('Geometry does not exist. Load it first with the `read` method.')
        for node in self.nodes:
            for index, coordinate in enumerate(node[1:], 1):
                node[index] = ' ' + str(float(coordinate)*factor)

    def write(self, output_file=None):
        """Writes material data to an Abaqus .inp file.

        Parameters
        ----------
        output_file : str, optional
            Output file name to write the modifications into.
            If not given, the original file name is appended with '_mod'.

        Returns
        -------
        None.

        Notes
        -----
        - If the output file name is the same as the input file, the original
          .inp file will be overwritten. This is strongly not recommended.
        - The whole content of the original input file is read to memory. It
          might be a problem for very large .inp files. In that case, a possible
          implementation could be the following:
              1. Remove old material data
              2. Append new material data to the proper position in the file
          Appending is automatically done at the end of the file. Moving the
          material description to the end of the file is not possible in general
          because defining materials cannot be done from any module, i.e. the
          *MATERIAL keyword cannot follow an arbitrary keyword. In this case,
          Abaqus throws an AbaqusException with the following message:
              It can be suboption for the following keyword(s)/level(s): model

        """
        if not self.nodes:
            raise Exception('Material does not exist. Nothing to write.')
        with open(self.inp_file, 'r') as source:
            old = source.readlines()
        before = old[0:self.state['begin']]
        after = old[self.state['end']+1:]
        new = before + self.__format() + after
        if not output_file:
            name, extension = os.path.splitext(self.inp_file)
            output_file = name + '_mod' + extension
        validate_file(output_file, 'write')
        with open(output_file, 'w') as target:
            target.write(''.join(new))
        print('Material database saved as `{0}`.'.format(output_file))

    def __format(self):
        """Formats the material data in the Abaqus .inp format.
        The internal representation of the material data in converted to a
        string understood by Abaqus.

        Returns
        -------
        abaqus_format : list
            List of strings, each element of the list corresponding to a line
            (with \n line ending) in the Abaqus .inp file. In case of no
            material, an empty list is returned.

        Notes
        -----
        The output is a list so that further concatenation operations are easy.
        If you want a string, merge the elements of the list:
            output = ''.join(output)
        This is what the `show` method does.

        """
        abaqus_format = ['*NODE\n']
        for node in self.nodes:
            abaqus_format.append(','.join(node) + '\n')
        return abaqus_format


def validate_file(file, caller):
    """Input or output file validation.

    Parameters
    ----------
    file : str
        Existing Abaqus .inp file or a new .inp file to be created.
    caller : {'read', 'write', 'create'}
        Method name that called this function.

    Returns
    -------
    None.

    """
    if type(file) is not str:
        raise Exception('String expected for file name.')
    file_extension = os.path.splitext(file)[1]
    if file_extension != '.inp':
        print('File extension `{0}` is not the conventional `.inp` '
              'used by Abaqus.'.format(file_extension))
    existing_file = os.path.isfile(file)
    if caller in {'read'}:
        if not existing_file:
            raise Exception('File does not exist.')
        empty_file = os.stat(file).st_size == 0
        if empty_file:
            raise Exception('Input file is empty.')
    elif caller in {'write', 'create'}:
        if existing_file:
            print('File `{0}` already exists. It will be overwritten!'.format(file))
    else:
        raise Exception('Only the following methods are allowed to call '
                        'this function: "read"", "write" "create".')

def extract(keyword):
    """Obtains Abaqus keyword and its parameters.
    Given a

    Parameters
    ----------
    keyword : str
        Some examples:
            '*Elastic, type=ORTHOTROPIC'
            '*Damage Initiation, criterion=HASHIN'
            '*Nset, nset=Set-1, generate'

    Returns
    -------
    separated : list
        DESCRIPTION.

    """
    # Split the string along the commas
    split = keyword[1:].split(',')
    # Remove leading and trailing whitespaces
    stripped = [x.strip() for x in split]
    # Separate parameters from values
    separated = [x.split('=') for x in stripped]
    return separated
