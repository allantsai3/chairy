import os
"""
This file contains the definition for a part
"""

class Part(object):
    def __init__(self, file_list, obj_file_loc):
        self.vertices = []
        self.faces = []

        self.load_part(file_list, obj_file_loc)
        # Get center of mass after loading all parts
        self.get_center_of_mass(self.vertices, self.faces)

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def get_center_of_mass(self, vertices, faces):
        pass

    def load_part(self, file_list, obj_file_loc):
        for file in file_list:
            # Extracted from https://inareous.github.io/posts/opening-obj-using-py
            try:
                f = open(os.path.join(obj_file_loc, file + ".obj"))
                for line in f:
                    if line[:2] == "v ":
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)

                        vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                        # vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                        self.vertices.append(vertex)

                    elif line[0] == "f":
                        string = line.replace("//", "/")
                        ##
                        i = string.find(" ") + 1
                        face = []
                        for item in range(string.count(" ")):
                            if string.find(" ", i) == -1:
                                face.append(string[i:-1])
                                break
                            face.append(string[i:string.find(" ", i)])
                            i = string.find(" ", i) + 1
                        ##
                        self.faces.append(tuple(face))

                f.close()
            except IOError:
                print(".obj file not found")


