import os
"""
This file contains the definition for a part
"""


class Part(object):
    def __init__(self, file_list, obj_file_loc, part_id = -1, log=False):
        self.vertices = []
        self.faces = []
        self.bounding_box = []
        self.part_id = part_id

        self.load_part(file_list, obj_file_loc)
        # Get center of mass after loading all parts
        self.get_center_of_mass(self.vertices, self.faces)

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def get_center_of_mass(self, vertices, faces):
        pass

    def get_bounding_box(self):
        return self.bounding_box

    def get_part_id(self):
        return self.part_id

    def set_bounding_box(self, new_bounding_box):
        self.bounding_box = new_bounding_box

    def load_part(self, file_list, obj_file_loc):
        # If each main part is composed of multiple parts
        for file in file_list:
            vertex_count = len(self.vertices)
            # Based on https://inareous.github.io/posts/opening-obj-using-py

            try:
                f = open(os.path.join(obj_file_loc, file + ".obj"))
                for line in f:
                    if line[:2] == "v ":
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)

                        vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                        vertex = (round(vertex[0], 4), round(vertex[1], 4), round(vertex[2], 4))

                        self.vertices.append(vertex)

                    elif line[0] == "f":
                        string = line.replace("//", "/")
                        ##
                        i = string.find(" ") + 1
                        face = []
                        for item in range(string.count(" ")):
                            if string.find(" ", i) == -1:
                                face.append(str(int(string[i:-1]) + vertex_count))
                                break
                            face.append(str(int(string[i:string.find(" ", i)]) + vertex_count))
                            i = string.find(" ", i) + 1
                        ##
                        self.faces.append(tuple(face))

                f.close()
            except IOError:
                print(".obj file not found")
