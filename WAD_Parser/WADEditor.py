import itertools
from struct import *
from WAD_Parser import Lumps
from WAD_Parser.WADFeatureExtractor import WADFeatureExtractor
import re, os
from skimage import io
from skimage.measure import find_contours
from skimage.measure import label
import subprocess
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from WAD_Parser.Dictionaries.ThingTypes import *
from WAD_Parser.RoomTopology import topological_features
from WAD_Parser.flags import linedef_flags_to_int
import networkx as nx
import warnings

from doomutils import vertices_to_segment_list
# Data specification taken from http://www.gamers.org/dhs/helpdocs/dmsp1666.html
# Implementation by Edoardo Giacomello Nov - 2017



class LumpInfo(dict):
    def __init__(self, filepos=None, size=None, name=None):
        """

        :param filepos:  An integer holding a pointer to the start of the lump's data in the file.
        :param size: An integer representing the size of the lump in bytes.
        :param name: A 8 byte encoded ascii string, eventually padded with 00
        """
        super()
        self['filepos'] = filepos
        self['size'] = size
        self['name'] = name


    def from_bytes(self, byte_stream):
        self['filepos'], = unpack("i", byte_stream[0:4])
        self['size'], = unpack("i", byte_stream[4:8])
        self['name'] = Lumps.decode_doomstring(byte_stream[8:16])
        return self

    def to_bytes(self):
        info_bytes = bytearray()
        info_bytes += pack("i", self['filepos'])
        info_bytes += pack("i", self['size'])
        info_bytes += Lumps.encode_doomstring(self['name'])
        return info_bytes
class WAD(dict):
    def __init__(self, mode):
        """
            Dictionary structured representation of a WAD file. Fields that are dictionary keys are unprocessed data from
            the file itself, while object attributes are "secondary access keys" (structures not directly encoded in the
             WAD file but built for faster access to data.)

             mode: 'R' for reading or 'W' for writing.

            Example:
                self['lumps'] contains the list of all the lumps.
                        Lumps describing levels are processed as list or dict(s), the others are kept as raw bytes.
                self['directory'] is the list of lump info, the structure reflects the one encoded in the WAD file.
                self.levels contains the lumps grouped by each level. (secondary key, not directly encoded into the file)
                self.sectors contains the linedefs sorrounding each map sector (secondary key)
            Warning:
                It's suggested to use the WADReader and WADWriter class in order to read and write a WAD.
                If you need to edit a WAD object, please consider copying it into another WAD() using from_bytes and
                to_bytes methods.
        """
        super()
        self['header'] = {
                'identification' : 'PWAD',  # Ascii identifier: IWAD or PWAD
                'numlumps' : 0, # An integer specifying the number of lumps in the WAD.
                'infotableofs' : 0 # An integer holding a pointer to the location of the directory.
            }
        self['lumps'] = []  # List of lumps, some processed, other in byte format
        self['directory'] = list() # List of lumpinfo
        self.levels = [] # this division in levels is not part of the wad but it's done for fast access
        self.map_regex = re.compile('MAP\d\d?')
        self.em_regex = re.compile('E\d*M\d\d?')
        self.errors = list()

        self.mode = mode
        self.current_lump_offset = 12  # Keeps track of the offset in bytes of the last. The header is always 12 bytes long



    def from_bytes(self, byte_stream):
        '''
        Builds a WAD object from the byte stream from a .WAD file.
        :param byte_stream:
        :return:
        '''
        assert self.mode == 'R', "Cannot read a WAD opened in write mode. " \
                                 "Please consider copying your WAD() into a new one " \
                                 "using to_bytes and from_bytes methods"
        try:
            self['header']['identification'] = Lumps.decode_doomstring(byte_stream[0:4])
            self['header']['numlumps'], = unpack("i", byte_stream[4:8])
            self['header']['infotableofs'], = unpack("i", byte_stream[8:12])

            # the pattern for grouped record is
            # [byte[start:start+length] for start in range(offset, offset+n_items*length, length)]
            lump_info_records = [byte_stream[start:start+16] for start in range(self['header']['infotableofs'],
                                                                                self['header']['infotableofs']
                                                                                +self['header']['numlumps']*16, 16)]
            # Populate the lump directory
            for lump_info_bytes in lump_info_records:
                lumpinfo = LumpInfo().from_bytes(lump_info_bytes)
                self['directory'].append(lumpinfo)

            # Parsing lumps
            for lump in self['directory']:
                if lump['size'] < 0:
                    self.errors.append({'object': lump, 'description': 'Negative size lump', 'fatal':False})
                    # Some files are corrupted and have a negative lump size. They'd cause a segfault if launched with doom
                    # We try to go on extracting as much data as we can from the WAD file.
                    continue
                lumpname = lump['name']
                # If the lump is a level descriptor, then create a new secondary key for the level
                if (self.map_regex.match(lump['name']) is not None) or (self.em_regex.match(lump['name']) is not None):
                    self.levels.append({'name':lumpname, 'lumps':{}})

                lump_bytes = byte_stream[lump['filepos']:lump['filepos'] + lump['size']]

                if lumpname in Lumps.known_lumps_classes.keys() and len(lump_bytes) > 0:
                    # Got a level lump and need to parse it...
                    l = Lumps.known_lumps_classes[lumpname]().from_bytes(lump_bytes)
                    if len(self.levels)>0:  #  otherwise we have found a level lump before the level description, which should not happen
                        self.levels[-1]['lumps'][lumpname] = l

                    # Adding processed lump to the lump list
                    self['lumps'].append(l)
                else:
                    # got an empty lump or another type of lump (such textures etc) that is not useful.
                    # Adding raw format to the lump list
                    self['lumps'].append(lump_bytes)

            # Cleaning empty levels (some wad files has random level descriptor with no lumps following them
            for l in self.levels:
                if 'SECTORS' not in l['lumps'] or 'LINEDEFS' not in l['lumps']:
                    self.levels.remove(l)


            # Building other secondary access keys
            # levels[sector][sector_id] = {sector: lump, sidedefs: list(lump), linedefs: list(lump), vertices=list(), vertex_path=list()}
            # Retrieving linedefs for each sector
            for level in self.levels:
                level['sectors'] = {}
                # This part of code makes the access to sectors and vertices easier.
                # Lines, Vertices, Sidedef and Sectors are indexed by three lists, and they are connected in this way:
                # Line -> Vertices, Line -> Sidedef(s) -> Sector

                # Create an entry for each sector.
                for sec_id, sec_lump in enumerate(level['lumps']['SECTORS']):
                    level['sectors'][sec_id] = {'lump': sec_lump, 'linedefs': list(), 'sidedefs': list(), 'vertex_path':list(), 'vertices_xy':list()}

                # For each linedef, fill the corresponding sector record(s)
                for linedef_id, linedef_lump in enumerate(level['lumps']['LINEDEFS']):
                    r_side_id = linedef_lump['right_sidedef']
                    r_sidedef = level['lumps']['SIDEDEFS'][r_side_id]
                    r_sector = r_sidedef['sector']
                    level['sectors'][r_sector]['linedefs'].append(linedef_lump)
                    level['sectors'][r_sector]['sidedefs'].append(r_sidedef)

                    l_side_id = linedef_lump['left_sidedef']
                    if l_side_id != -1:
                        l_sidedef = level['lumps']['SIDEDEFS'][l_side_id]
                        l_sector = l_sidedef['sector']
                        level['sectors'][l_sector]['linedefs'].append(linedef_lump)
                        level['sectors'][l_sector]['sidedefs'].append(l_sidedef)

                # create vertex path for each sector for drawing
                for sector_id, sector in level['sectors'].items():
                    # Make the graph G(Linedefs, Verices) undirected
                    edges = set()
                    for linedef in sector['linedefs']:
                        if (linedef['from'] != linedef['to']):  # Avoids single-vertex linedefs
                            edges.add((linedef['from'],linedef['to']))
                            edges.add((linedef['to'],linedef['from']))
                    if len(edges) > 0:   # Avoid crashes if some sectors are empty
                        # "hops" is the list of vertex indices as visited by a drawing algorithm
                        hops = list()
                        next_edge = min(edges)

                        if next_edge[0] not in hops:
                            hops.append(next_edge[0])
                        if next_edge[1] not in hops:
                            hops.append(next_edge[1])
                        while (len(edges) > 1):
                            edges.remove((next_edge[1], next_edge[0]))
                            edges.remove((next_edge[0], next_edge[1]))
                            next_edges = set(filter(lambda x: x[0] == hops[-1] or x[1] == hops[-1], edges))
                            if len(next_edges) == 0:
                                break
                            possible_next = min(next_edges)
                            if possible_next[1] == hops[-1]:
                                next_edge = (possible_next[1], possible_next[0])
                            else:
                                next_edge = possible_next
                            if next_edge[-1] not in hops:
                                hops.append(next_edge[-1])
                        sector['vertex_path'] = hops
                        sector['vertices_xy'] = [(level['lumps']['VERTEXES'][v_id]['x'], level['lumps']['VERTEXES'][v_id]['y']) for v_id in hops]
        except Exception as e:
            # All known exceptions found in the database are avoided, this exception will catch everything else that is unexpected
            # and will produce a fatal error
            self.errors = list()
            self.errors.append({'object': self, 'description': e, 'fatal':True})

        return self

    def add_lump(self, lumpname, lump):
        """
        Adds a new lump named lumpname and updates the information in the directory. Increments the current_lump_offset.
        :param lumpname: lump name. It will be converted in doomstring format.
        :param lump: a @Lumps object, or None for level descriptors or other zero-sized lumps.
        :return: None
        """
        assert self.mode == 'W', "Cannot write a WAD opened in read mode. " \
                                 "Please consider copying your WAD() into a new one " \
                                 "using to_bytes and from_bytes methods"
        if lump is None:
            lump_bytes = bytes()
        else:
            lump_bytes = lump.to_bytes()
        size = len(lump_bytes)
        self['directory'].append(LumpInfo(filepos=self.current_lump_offset, size=size, name=lumpname))
        self['lumps'].append(lump_bytes)
        # Updating directory and header information
        self.current_lump_offset += size
        self['header']['numlumps'] += 1
        # The infotableoffset is always kept at the end of the file
        self['header']['infotableofs'] = self.current_lump_offset

    def to_bytes(self):
        # Build the entire file
        # header to bytes
        wad_bytes = bytearray()
        wad_bytes += bytes('PWAD', encoding='ascii')
        wad_bytes += pack('i', self['header']['numlumps'])
        wad_bytes += pack('i', self['header']['infotableofs'])

        # Adding Lumps
        for lump in self['lumps']:
            wad_bytes += lump

        # Adding directory
        for lumpinfo in self['directory']:
            wad_bytes += lumpinfo.to_bytes()
        return wad_bytes


class WADWriter(object):
    def __init__(self):
        """
        Class for writing a WAD file.
        Start by defining a new level with add_level(), then place new sectors and "things". Changes are submitted only
         on save or on the addition of a new level, since some optimization are due (duplicated vertices check, etc).
        """
        # due to the way the WAD works, for keeping track of the lumps size for filling the dictionary the byte representation is
        # needed. But for checking duplicated vertices/linedefs/etc it would be needed to convert back each lump before the
        # check. For avoiding this problem, a set of lumps is stored in the writer and written only when the level is
        # fully specified.
        self.wad = WAD('W')
        self.current_level = None
        self.lumps = {'THINGS':Lumps.Things(), 'LINEDEFS':Lumps.Linedefs(), 'VERTEXES':Lumps.Vertexes(),'SIDEDEFS': Lumps.Sidedefs(), 'SECTORS':Lumps.Sectors()}  # Temporary lumps for this level

    def _sector_orientation(self, vertices):
        """
        Check if the polygon is oriented clock-wise or counter-clockwise.
        If the polygon is not closed, then closes it
        :param vertices: the input vertices
        :return: (Bool, vertices) True if clockwise, False otherwise.
        """
        if not vertices[0] == vertices[-1]:
            vertices.append(vertices[0])
        xy = np.transpose(np.array(vertices))
        x, y = xy[0], xy[1]
        return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0, vertices

    def _rescale_coords(self, v, scale_factor=64):
        return tuple(int(a * scale_factor) for a in v)

    def from_images_v2(self, floormap, heightmap, wallmap, thingsmap):
        from skimage.io import imshow
        if isinstance(floormap, str):
            floormap = io.imread(floormap).astype(dtype=np.bool_)
        if isinstance(heightmap, str):
            heightmap = io.imread(heightmap).astype(np.uint8)
        if isinstance(wallmap, str):
            wallmap = io.imread(wallmap).astype(dtype=np.bool_)
        if isinstance(thingsmap, str):
            thingsmap = io.imread(thingsmap).astype(np.uint8)

        walkable = np.logical_and(floormap, np.logical_not(wallmap)) if wallmap is not None else floormap

        walkable = morphology.remove_small_objects(walkable)
        walkable = morphology.remove_small_holes(walkable)

        roommap, graph, metrics = topological_features(walkable, prepare_for_doom=True)
        graph = self.decorate_graph(graph, roommap, heightmap, thingsmap)

        self.from_graph(graph)

        # Set the start somewhere in room 1
        # TODO: Implement data from graph
        for n in graph.nodes():
            if n == 0:
                continue
            spot = tuple(np.floor(graph.node[n]["centroid"]).astype(np.int))
            if roommap[spot] != 0:
                x,y = self._rescale_coords(spot)
                self.set_start(x,y)
                break

    def decorate_graph(self, G, roommap, heightmap, thingsmap):
        """
        Adds information about the heightmap and the thingsmap in the region adjacency graph.
        :param roommap:
        :param heightmap:
        :param thingsmap:
        :return: updated G
        """
        # Selecting candidates for starting and exiting nodes:
        # leaves of the spanning tree are the most suitable
        # Connected components (floors)

        H = G.copy()
        H.remove_node(0)
        floors = sorted(nx.connected_components(H), key=len, reverse=True)
        level_solution = list()
        corrected_heights = dict()

        for id, floor_rooms in enumerate(floors):
            # Creating a spanning tree for each floor
            F = H.subgraph(floor_rooms)
            T = nx.minimum_spanning_tree(F)
            degree = T.degree()
            # Entry point has minimum node degree
            floor_entry = min(degree, key=degree.get)
            # Finding all paths in the level to determine the best exit (longest path)
            paths = list()
            for n in T.nodes():
                paths += nx.all_simple_paths(T, floor_entry, n)
            floor_solution = max(paths, key=len)
            level_solution.append(floor_solution)

            # Fixing the heights along all paths so every path becomes walkable
            for path in paths:
                for rid, room in enumerate(path):
                    if room not in corrected_heights:
                        height = np.nanmedian(np.where(roommap == room, heightmap, np.nan))
                        if rid > 0:
                            # Alter this room height to be walkable
                            if height > path[rid-1] + 24:
                                height = path[rid-1] + 24
                        corrected_heights[room] = int(height)
        nx.set_node_attributes(G, "height", corrected_heights)

        for id, floor_path in enumerate(level_solution):
            if id == 0:
                # Place the level start
                nx.set_node_attributes(G, "level_start", {floor_path[0]: True})
            else:
                # place a teleport dest
                nx.set_node_attributes(G, "floor_start", {floor_path[0]: True})
            if id == len(level_solution)-1:
                # This is the last floor to visit, place the level exit
                nx.set_node_attributes(G, "level_exit", {floor_path[-1]: True})
            else:
                # There's another unvisited floor, place a teleporter to the next floor
                nx.set_node_attributes(G, "floor_exit", {floor_path[-1]: level_solution[id+1][0]})

        return G

    def from_graph(self, graph):
        """
        Builds a level exploiting information stored in the room adjacency graph. Treat each room as a different sector.
        :param graph:
        :return:
        """
        edge_attr_sidedef = dict()  # Dictionary for updating edge attributes
        node_attr_sectors = dict()  # Dictionary for updating edge attributes
        # Creating a sector for each room
        for n in graph.nodes():
            if n == 0:
                continue
            heights = nx.get_node_attributes(graph, "height")
            # Create a sector
            node_attr_sectors[n] = self.lumps['SECTORS'].add_sector(floor_height=int(heights[n]), ceiling_height=128+int(heights[n]), floor_flat='FLOOR0_1', ceiling_flat='FLOOR4_1', lightlevel=255,
                                                         special_sector=0, tag=int(n))
        nx.set_node_attributes(graph, 'sector_id', node_attr_sectors)

        # Creating two sidedefs for each edge and the corresponding linedef
        for i, j in graph.edges():
            if i == 0:
                # linedef flag is impassable and the right sidedef is j
                j_walls = graph.node[j]["walls"]
                walls = [w for w in j_walls if w[1] == i or w[1] is None] # Check this if there's any problem in the corners
                for wall_piece in walls:
                    start = self.lumps['VERTEXES'].add_vertex(self._rescale_coords(wall_piece[0][0]))
                    end = self.lumps['VERTEXES'].add_vertex(self._rescale_coords(wall_piece[0][1]))
                    # - Right sidedef is j
                    right_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0,
                                                                      upper_texture='BRONZE1',
                                                                      lower_texture='BRONZE1',
                                                                      middle_texture='BRONZE1',
                                                                      sector=graph.node[j]["sector_id"])
                    # - Make a linedef
                    linedef = self.lumps['LINEDEFS'].add_linedef(start, end, flags=linedef_flags_to_int(impassable=True), types=0,
                                               trigger=0, right_sidedef_index=right_sidedef)
                    # Save the linedef into the edge
                    if (i,j) not in edge_attr_sidedef:
                        edge_attr_sidedef[(i,j)] = list()
                    edge_attr_sidedef[(i, j)].append(right_sidedef)
            else:
                i_walls = graph.node[i]["walls"]
                # linedef is invisible
                # Get the boundaries from i to j
                walls_ij = [w for w in i_walls if w[1] == j]
                for wall_piece in walls_ij:
                    start = self.lumps['VERTEXES'].add_vertex(self._rescale_coords(wall_piece[0][0]))
                    end = self.lumps['VERTEXES'].add_vertex(self._rescale_coords(wall_piece[0][1]))
                    # - Right sidedef is i
                    right_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0,
                                                                      upper_texture='BRONZE1',
                                                                      lower_texture='BRONZE1',
                                                                      middle_texture='-',
                                                                      sector=graph.node[i]["sector_id"])
                    # - Left sidedef is j (in j list there's the reversed linedef)
                    left_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0,
                                                                      upper_texture='BRONZE1',
                                                                      lower_texture='BRONZE1',
                                                                      middle_texture='-',
                                                                      sector=graph.node[j]["sector_id"])
                    # - Make a linedef
                    linedef = self.lumps['LINEDEFS'].add_linedef(start, end, flags=linedef_flags_to_int(twosided=True), types=0,
                                               trigger=0, right_sidedef_index=right_sidedef,
                                               left_sidedef_index=left_sidedef)
                    # Save the linedef into the edge
                    if (i, j) not in edge_attr_sidedef:
                        edge_attr_sidedef[(i, j)] = list()
                    edge_attr_sidedef[(i, j)].append(right_sidedef)
                    edge_attr_sidedef[(i, j)].append(left_sidedef)
            # Actually update edge attribnutes
            nx.set_edge_attributes(graph, 'sidedef', edge_attr_sidedef)



    def from_images(self, heightmap, floormap, wallmap, thingsmap=None, level_coord_scale = 64, debug = False, save_debug = './generated_levels_debug/', generate_teleporters=True, flat_levels=False):
        assert heightmap is not None or floormap is not None, "heightmap and floormap cannot be both None"

        if isinstance(floormap, str):
            floormap = io.imread(floormap).astype(dtype=np.bool_)
        if isinstance(heightmap, str):
            heightmap = io.imread(heightmap).astype(np.uint8)
        if isinstance(wallmap, str):
            wallmap = io.imread(wallmap).astype(dtype=np.bool_)
        if isinstance(thingsmap, str):
            thingsmap = io.imread(thingsmap).astype(np.uint8)

        if thingsmap is not None:
            # Pad with a frame
            thingsmap = np.pad(thingsmap, pad_width=(1, 1), mode='constant')

        if floormap is None:
            floormap = heightmap > 0
        if heightmap is None:
            heightmap = np.ones_like(floormap) * 128

        # Pad with a frame for getting boundaries
        floormap = np.pad(floormap, pad_width=(1, 1), mode='constant').astype(np.uint8)
        wallmap = np.pad(wallmap, pad_width=(1, 1), mode='constant').astype(np.uint8)
        heightmap = np.pad(heightmap, pad_width=(1, 1), mode='constant').astype(np.uint8)

        floormap = morphology.binary_dilation(floormap)

        # Apply some morphology transformation to the maps for combining them and removing artifacts
        denoised = morphology.remove_small_holes(floormap, min_size=16)
        denoised = morphology.remove_small_objects(denoised, min_size=16)
        denoised_walls = np.logical_and(denoised, np.logical_not(wallmap))
        cleaned_walls = morphology.remove_small_holes(denoised_walls, min_size=4)
        cleaned_walls = morphology.remove_small_objects(cleaned_walls, min_size=16)

        # HEIGHTMAP
        from skimage import data, util, filters, color
        from skimage.morphology import watershed
        from skimage.segmentation import slic, slic_superpixels
        from scipy.ndimage import median
        edges = filters.sobel(heightmap)
        grid = util.regular_grid(heightmap.shape, n_points=300)
        seeds = np.zeros(heightmap.shape, dtype=int)
        seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
        segmented_heightmap = watershed(edges, seeds)

        label_medians = median(heightmap, segmented_heightmap, index=range(1, segmented_heightmap.max() + 1))

        for l in range(1, segmented_heightmap.max() + 1):
            segmented_heightmap = np.where(segmented_heightmap == l, label_medians[l - 1], segmented_heightmap)


        if debug:
            # Display the image and plot all contours found
            fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 5), sharex=True,
                                   sharey=True, subplot_kw={'adjustable': 'box-forced'})
        if save_debug:
            fig_show_levels = plt.subplot()

        # Placing sectors, one floor at a time

        from scipy.ndimage.measurements import label
        segmented, total_floors = label(cleaned_walls, structure=np.ones((3, 3)))
        floors = [np.equal(segmented, f) for f in range(1, total_floors)]
        teleport_graph = np.roll(range(len(floors)), -1).tolist()
        for floor_id, floor in enumerate(floors):
            contours_walls = find_contours(floor, 0.5, positive_orientation='low')
            for i, floor_contour in enumerate(contours_walls):
                if debug:
                    ax[4].plot(floor_contour[:, 1], floor_contour[:, 0], linewidth=1)
                if save_debug:
                    fig_show_levels.plot(floor_contour[:, 1], floor_contour[:, 0], linewidth=1)
                vertices = (floor_contour * level_coord_scale).astype(np.int).tolist()
                clockwise, vertices = self._sector_orientation(vertices)
                if clockwise:
                    # The contour defines the floor boundaries
                    sector_id = self.add_sector(vertices, ceiling_height=128, tag=floor_id)
                else:
                    if i ==0:
                        sector_id = self.add_sector(list(reversed(vertices)), ceiling_height=128, tag=floor_id)

                    #if i == 0:
                    #    # The first sector has been defined as counter-clockwise, need a bounding sector
                    #    level_height = floormap.shape[0]*level_coord_scale
                    #    level_width = floormap.shape[1]*level_coord_scale
                    #    sector_id = self.add_sector([(0,0),(0,level_height),(level_width,level_height),(level_width, 0)], floor_height=-255*level_coord_scale, ceiling_height=255*level_coord_scale, tag=floor_id)
                    #    print("Added a bounding box because the first sector has been specified as counter-clockwise")
                    # The contour defines a hole in the level: Surrounding sector must be defined explicitly
                    self.add_sector(vertices, ceiling_height=128, tag=floor_id, surrounding_sector_id=sector_id, hollow=True)
                    # sector_id is not changed because we don't want to place anything inside a hole, hopefully.
            if not flat_levels:
                # Placing inner sectors (heightmap)
                floor_heightmap = segmented_heightmap*floor
                for sector_height in np.unique(floor_heightmap):
                    if sector_height == 0:
                        continue
                    same_height_map = (floor_heightmap == sector_height)*1
                    sector_contours = find_contours(same_height_map, 0.5, positive_orientation='low')
                    for ch_number, sector_contours in enumerate(sector_contours):
                        vertices = (sector_contours * level_coord_scale).astype(np.int).tolist()
                        clockwise, vertices = self._sector_orientation(vertices)
                        if not clockwise:
                            vertices = list(reversed(vertices))
                        # Drawing the sector with invisible walls
                        self.add_sector(vertices, floor_height=int(sector_height*level_coord_scale), ceiling_height=int(sector_height*level_coord_scale)+256,
                                        kw_sidedef={'upper_texture':'BRONZE1', 'lower_texture':'BRONZE1', 'middle_texture':'-'},
                                        surrounding_sector_id=sector_id,
                                        kw_linedef={'type': 0, 'trigger': 0, 'flags': 4})

            # Placing teleporters
            if total_floors > 1 and generate_teleporters:
                # Teleporters are needed. Place a landing somewhere in the sector
                possible_coords = np.transpose(np.nonzero(floor))
                rand_indices = tuple(np.random.choice(range(len(possible_coords)), size=2, replace=False).tolist())
                dest_coord = possible_coords[rand_indices[0]] * level_coord_scale
                src_coord = possible_coords[rand_indices[1]] * level_coord_scale
                # Creating a single loop visiting each floor exactly once (Eulerian cycle) in a random order
                dest_sector = teleport_graph[floor_id]
                # Add a destination in this sector
                self.add_teleporter_destination(dest_coord[0], dest_coord[1])
                # Add a source for another sector
                self.add_teleporter_source(src_coord[0], src_coord[1], dest_sector, inside=sector_id)



            # Finding and placing new player start
            possible_pos = np.where(np.logical_and(np.equal(thingsmap, 1), cleaned_walls))
            if (len(possible_pos[0]) == 0):
                possible_pos = np.where(cleaned_walls)
            rand_i = np.random.choice(range(len(possible_pos[0])))
            start_pos = (
            int(possible_pos[0][rand_i]) * level_coord_scale, int(possible_pos[1][rand_i]) * level_coord_scale)
            self.set_start(start_pos[0], start_pos[1])

            # Placing "Things"
            if thingsmap is not None:
                floor_things = floor * thingsmap
                for x,y in np.transpose(np.nonzero(floor_things)):
                    # FIXME: When things are put, player cannot move
                    # FIXME: When adding triggers, check for the sector tag cause currently is the floor tag

                    # Pixels are in range (0,255) but things type are in range (0,122). Moreover, 0 means "no thing"
                    pixel = int(floor_things[x, y])
                    # feat_scaling = lambda x, min, max, a, b: np.around(a + ((x-min)*(b-a))/(max-min))
                    # type_index = feat_scaling(pixel, 1, 255, 0,122)
                    type = get_type_id_from_index(pixel)
                    too_close_to_start = abs(x - start_pos[0]) < 64 * level_coord_scale or abs(y - start_pos[1]) < 64 * level_coord_scale
                    if type is not None and get_category_from_type_id(type) != 'start' and not too_close_to_start:
                        self.add_thing(level_coord_scale*x,level_coord_scale*y, type)

        if debug:
            ax[0].imshow(floormap, cmap=plt.cm.gray)
            ax[1].imshow(wallmap, cmap=plt.cm.gray)
            ax[2].imshow(denoised, cmap=plt.cm.gray)
            ax[3].imshow(denoised_walls, cmap=plt.cm.gray)
            ax[4].imshow(cleaned_walls, cmap=plt.cm.gray)
            ax[4].imshow(segmented, cmap=plt.cm.gray)
        if debug:
            plt.show()
        if save_debug:
            plt.axis('off')
            fig_show_levels.imshow(segmented, cmap=plt.cm.gray)
            os.makedirs(save_debug, exist_ok=True)
            plt.savefig(save_debug+'lv_{}.png'.format(self.current_level), bbox_inches='tight')
        plt.clf()



    def add_teleporter_destination(self, x, y):
        self.add_thing(x, y, thing_type=14)

    def add_teleporter_source(self, x, y, to_sector, inside, size=32):
        x=int(x)
        y=int(y)
        to_sector=int(to_sector)
        halfsize=size//2
        vertices = list(reversed([(x-halfsize, y+halfsize),(x+halfsize, y+halfsize),(x+halfsize, y-halfsize),(x-halfsize, y-halfsize)]))
        self.add_sector(vertices, floor_flat='GATE1', kw_sidedef={'upper_texture':'-', 'lower_texture':'-', 'middle_texture':'-'}, kw_linedef={'flags':4, 'type':97, 'trigger': to_sector}, surrounding_sector_id=inside)



    def set_start(self, x, y):
        self.lumps['THINGS'].add_thing(int(x), int(y), angle=0, type=1, options=0)

    def add_thing(self, x,y, thing_type, options=7, angle=0):
        self.lumps['THINGS'].add_thing(int(x), int(y), angle=angle, type=int(thing_type), options=options)

    def add_door(self,vertices_coords, parent_sector, tag=None, remote=False, texture='DOORTRAK'):
        """
        adds a door with a given tag. If tag is left unspecified, then it will be equal to the sector index.
        :param vertices_coords:
        :param parent_sector:
        :param tag:
        :param remote:
        :param texture:
        :return:
        """
        height = self.lumps['SECTORS'][parent_sector]['floor_height']
        type = 1 if not remote else 0
        tag = len(self.lumps['SECTORS']) if tag is None else tag
        return self.add_sector(vertices_coords, ceiling_height=height, kw_sidedef={'upper_texture':texture, 'lower_texture':texture, 'middle_texture':'-'}, kw_linedef={'type':type, 'flags':4, 'trigger':0}, tag=tag, surrounding_sector_id=parent_sector, hollow=False)

    def add_trigger(self, vertices_coords, parent_sector, trigger_type, trigger_tag, texture='SW1CMT'):
        return self.add_sector(vertices_coords,
                               kw_sidedef={'upper_texture': 'BRONZE1', 'lower_texture': 'BRONZE1', 'middle_texture': texture},
                               kw_linedef={'type': trigger_type, 'flags': 1, 'trigger': trigger_tag}, tag=0,
                               surrounding_sector_id=parent_sector, hollow=False)

    def add_sector(self, vertices_coords, floor_height=0, ceiling_height=128, floor_flat='FLOOR0_1', ceiling_flat='FLOOR4_1', lightlevel=256, special=0, tag=0, surrounding_sector_id=None, hollow=False, kw_sidedef=None, kw_linedef=None):
        """
         Adds a sector with given vertices coordinates, creating all the necessary linedefs and sidedefs and return the relative
        sector id for passing the reference to other sectors or objects if needed
        :param vertices_coords: Vertices coordinates (x,y),(x2,y2).. If given in CLOCKWISE order then the room will have
        its right linedefs facing INWARD, the left one can be left unspecified (i.e. sorrounding_sector_id = None e.g. for the outermost sector)
        and the hollow parameter has no effect.
          If vertices are in COUNTER-CLOCKWISE order, then you are defining a sector with RIGHT SIDEDEFS facing outside,
          for that reason the sorrounding_sector_id parameter is mandatory and you are creating a sector inside another sector,
          like a column, a wall or a door. You can set if the linedefs can contain an actual sector or not with the "hollow" parameter.
        :param floor_height: height of the floor in doom map units
        :param ceiling_height:
        :param floor_flat:
        :param ceiling_flat:
        :param lightlevel:
        :param special:
        :param tag:
        :param wall_texture:
        :param surrounding_sector_id: sector id (returned by this function itself) for the sector that surrounds the one you are creating. Can be None only if the vertices are specified in clockwise order, since a linedef must have a sector on its right side.
        :param hollow: Has effect only for counter-clockwise specified sectors. Determines if the sector you are creating does actually contains a sector (like for doors) or it's just a hole surrounded by walls/linedefs, like the column or other static structures. Default to False.
        :param kw_linedef: (Optional) A list of dictionaries containing the parameters for each linedef, or a single dictionary if all linedefs share the same parameters. Must have the indices: 'type', 'trigger' and 'flags'. Default: {'type':0, 'flags':17, 'trigger':0}
        :return:
        """
        # In order to add a sector one must add:
        # A vertex for each vertex in the sector, only if not already present
        # a linedef for each edge, if not already present
        # a sidedef for each linedef, put on the correct side of the linedef
        # the sector lump itself

        # Create a sector
        sector_id = self.lumps['SECTORS'].add_sector(floor_height, ceiling_height, floor_flat, ceiling_flat, lightlevel, special, tag)


        # Create and lookup vertices
        vertices_id = list()
        for v in vertices_coords:
            v_id = self.lumps['VERTEXES'].add_vertex(v)
            vertices_id.append(v_id)
        clockwise, _ = self._sector_orientation(vertices_coords)
        # Iterate over (v0, v1), (v1, v2), ..., (vn-1, vn).
        # Adding the first element at the end for closing the polygon
        startiter, enditer = itertools.tee(vertices_id+[vertices_id[0]], 2)
        next(enditer, None)  # shift the end iterator by one
        for segcounter, (start, end) in enumerate(zip(startiter, enditer)):
            if kw_sidedef is None:
                kw_sidedef = {'upper_texture':'-', 'lower_texture':'-', 'middle_texture':'BRONZE1'}
            if clockwise:
                # The room has the right sidedef facing surrounding_sector_id, toward the new sector
                right_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0, upper_texture=kw_sidedef['upper_texture'], lower_texture=kw_sidedef['lower_texture'],
                                                   middle_texture=kw_sidedef['middle_texture'], sector=sector_id)
                if surrounding_sector_id is not None:
                    left_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0, upper_texture=kw_sidedef['upper_texture'], lower_texture=kw_sidedef['lower_texture'], middle_texture=kw_sidedef['middle_texture'], sector=surrounding_sector_id)
                else:
                    left_sidedef=-1
            else:
                # The room has the right sidedef facing outside, towards the sorrounding sector
                right_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0,
                                                                   upper_texture=kw_sidedef['upper_texture'],
                                                                   lower_texture=kw_sidedef['lower_texture'],
                                                                   middle_texture=kw_sidedef['middle_texture'],
                                                                   sector=surrounding_sector_id)
                if not hollow:
                    left_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0,
                                                                      upper_texture=kw_sidedef['upper_texture'],
                                                                      lower_texture=kw_sidedef['lower_texture'],
                                                                      middle_texture=kw_sidedef['middle_texture'],
                                                                      sector=sector_id)
                else:
                    left_sidedef = -1

            # Linedef creation/Lookup
            if kw_linedef is None:
                linedef_params = {'type':0, 'flags':17, 'trigger':0}
            elif isinstance(kw_linedef, dict):
                linedef_params = kw_linedef
            elif isinstance(kw_linedef, list):
                linedef_params = kw_linedef[segcounter]
            else:
                raise ValueError("kw_linedef can only be None, a Dict or a list of Dict.")
            self.lumps['LINEDEFS'].add_linedef(start, end, flags=linedef_params['flags'], types=linedef_params['type'], trigger=linedef_params['trigger'], right_sidedef_index=right_sidedef, left_sidedef_index=left_sidedef)
        return sector_id

    def _commit_level(self):
        """
        Writes the set of lumps to the WAD object
        :return:
        """
        assert self.current_level is not None, "Cannot write a level with an empty name"
        # Create a new level descriptor in the lump directory
        self.wad.add_lump(self.current_level, None)
        # Add the lumps to WAD file
        self.wad.add_lump('THINGS', self.lumps['THINGS'])
        self.wad.add_lump('LINEDEFS', self.lumps['LINEDEFS'])
        self.wad.add_lump('SIDEDEFS', self.lumps['SIDEDEFS'])
        self.wad.add_lump('VERTEXES', self.lumps['VERTEXES'])
        self.wad.add_lump('SECTORS', self.lumps['SECTORS'])
        self.lumps = {'THINGS':Lumps.Things(), 'LINEDEFS':Lumps.Linedefs(), 'VERTEXES':Lumps.Vertexes(),'SIDEDEFS': Lumps.Sidedefs(), 'SECTORS':Lumps.Sectors()}

    def add_level(self, name='MAP01'):
        # If it's not the first level, then commit the previous
        if self.current_level is not None:
            self._commit_level()
        self.current_level = name

    def save(self, fp, call_node_builder=True):
        # Always commit the last level
        self._commit_level()
        wad_bytes = self.wad.to_bytes()
        with open(fp,'wb') as out:
            out.write(wad_bytes)
        if call_node_builder:
            # Calling ZenNode to build subsectors and other lumps needed to play the level on doom
            print('Calling bsp...')
            subprocess.check_call(["bsp", fp, '-o', fp] )



class WADReader(object):
    """"Batch reader for WAD files"""

    def read(self, w):
        """
        Reads a wad file representing it as a dictionary
        :param w: path of the .WAD file
        :return:
        """
        with open(w, 'rb') as file:
            wad_name = w.split('/')[-1]
            wad = WAD('R').from_bytes(file.read())
            record = {'wad_name': wad_name, 'wad': wad, 'errors':list()}
            if len(record['wad'].errors) > 0:
                if (not record['wad'].errors[0]['fatal']):
                    print("{}: Malformed file, results may be altered".format(w))
                else:
                    print("{}: Fatal error in file structure, skipping file.".format(w))
                    record['errors'] += record['wad'].errors
        return record

    def save_sample(self, wad, path, root_path = '', wad_info=None):
        """
        Saves the wad maps (as .png) and features (as .json) to the "path" folder for each level in the wad.
        Also adds the produced file paths to the level features,
        if root_path is set then these paths are relative to that folder instead of being absolute paths
        if wad_info is not None, then adds its fields as features
        :param wad: the parsed wad file to save as feature maps
        :param path: the output folder
        :param root_path: the path to which the paths stored in the features should be relative to
        :return: None
        """
        os.makedirs(path, exist_ok=True)
        for level in wad['levels']:
            base_filename=path+wad['wad_name'].split('.')[-2]+'_'+level['name']
            # Path relative to the dataset root that will be stored in the database
            relative_path = base_filename.replace(root_path, '')
            # Adding the features
            for map in level['maps']:
                # Adding the corresponding path as feature for further access
                level['features']['path_{}'.format(map)] = relative_path + '_{}.png'.format(map)
                io.imsave(base_filename + '_{}.png'.format(map), level['maps'][map])
            for wadinfo in wad_info:
                # Adding wad info (author, etc) to the level features.
                if wadinfo not in level['features']:  # Computed features have priority over provided features
                    level['features'][wadinfo] = wad_info[wadinfo]
            # Completing the features with the level slot
            level['features']['slot'] = level['name']
            # Doing the same for the other features
            level['features']['path_json'] = relative_path + '.json'
            with open(base_filename + '.json', 'w') as jout:
                json.dump(level['features'], jout)
            # Saving the text representation
            with open(base_filename + '.txt', 'wb') as txtout:
                txtout.writelines([bytes(row + [10]) for row in level['text']])
            # Saving the graph
            if 'graph' in level:
                with open(base_filename + '.networkx', 'wb') as graphout:
                    nx.write_gpickle(level['graph'], graphout)


    def extract(self, wad_fp, save_to=None, root_path=None, update_record=None):
        """
        Compute the image representation and the features of each level contained in the wad file.
        If 'save_to' is set, then also do the following:
            - saves a json file for each level inside the 'save_to' folder
            - saves the set of maps as png images inside the 'save_to' folder
            - adds the relative path of the above mentioned files as level features
        Morover, if 'save_to' is set, then you may want to specify a 'root_path' for avoiding to save the full path in the features.
        If 'update_record' is set to a json dictionary (perhaps containing info about the wad author, title, etc),
        then this function stores all the update_record fields into the level features dictionary.
        :return: Parsed Wad
        """
        parsed_wad = self.read(wad_fp)
        for error in parsed_wad['errors']:
            if error['fatal']:
                return None
        parsed_wad['levels'] = list()
        for level in parsed_wad['wad'].levels:
            try:
                extractor = WADFeatureExtractor()
                features, maps, txt, graph = extractor.extract_features_from_wad(level)
                parsed_wad['levels'] += [{'name': level['name'], 'features': features, 'maps':maps, 'text':txt, 'graph':graph}]
                print("Extracted data for level {}".format(level['name']))
            except Exception as e:
                print("Error extracting data for level {}: {}".format(level['name'], e))
                traceback.print_exc()
            self.save_sample(parsed_wad, save_to, root_path, update_record)
        return parsed_wad


