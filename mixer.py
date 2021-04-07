import sys
import os
import random
import json
from Part import Part
"""
Axioms for chair:
 - Something that can be sit on (has a base)
 - center of balance/ stable (unless rocking chair etc..)
"""

"""
    Input: set of 3D parts and labels
    Output: Generated 3D model (and also orthogonal projection in multiple views)
"""


def reconstruct_obj(part_list):
    # Create the output directory if it doesn't exist
    directory = "tmp_output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open("./tmp_output/output_mesh.obj", "w")

    vertex_count = 0
    for num, part in enumerate(part_list):
        f.write("g " + str(num) + "\n")
        vertices = part.get_vertices()
        
        for vertex in vertices:
            f.write("v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + "\n")

        faces = part.get_faces()
        for face in faces:
            f.write("f " + str(int(face[0]) + vertex_count) + " "
                    + str(int(face[1]) + vertex_count) + " "
                    + str(int(face[2]) + vertex_count) + "\n")

        vertex_count += len(vertices)


if __name__ == "__main__":
    dataDir = sys.argv[1]

    sub_folders = [name for name in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, name))]

    # random.seed(23)
    index = random.choices(range(len(sub_folders)), k=4)

    chosenParts = [sub_folders[part] for part in index]

    print("Chosen parts are from folder:")
    print(chosenParts)

    partsList = []

    # Choose the chair_seat, chair_back, chair_base, chair_arm
    for i, part in enumerate(chosenParts):
        with open(os.path.join(dataDir, part, "result_after_merging.json")) as json_file:
            try:
                data = json.load(json_file)
            except ValueError:
                print(part)
            except IOError:
                print('IOError')

            obj_Files = os.path.join(dataDir, part, "objs")

            for obj in data[0]['children']:
                partName = obj['name']
                if partName == 'chair_seat' and i == 0:
                    partsList.append(Part(obj["objs"], obj_Files))

                if partName == 'chair_back' and i == 1:
                    partsList.append(Part(obj["objs"], obj_Files))

                if partName == 'chair_base' and i == 2:
                    partsList.append(Part(obj["objs"], obj_Files))

                if partName == 'chair_arm' and i == 3:
                    partsList.append(Part(obj["objs"], obj_Files))

    # Combine the various part vertices/faces and then output obj files
    reconstruct_obj(partsList)

