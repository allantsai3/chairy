import sys
import os
import random
import json
import cv2
from Part import Part
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import geometric_helpers

"""
Axioms for chair:
 - Something that can be sit on (has a base)
 - center of balance/ stable (unless rocking chair etc..)
"""

"""
    Input: set of 3D parts and labels
    Output: Generated 3D model obj file
"""


def plot_bounding_box(bounding_box1, bounding_box2):
    i1 = np.array(bounding_box1)
    x1, y1, z1 = i1.T

    i2 = np.array(bounding_box2)
    x2, y2, z2 = i2.T

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.plot(x1, y1, z1, color='blue')
    ax.plot(x2, y2, z2, color='C1')
    plt.show()


def update_part_vertices(part, part_bounding_box, ref_part_bounding_box):
    t = geometric_helpers.bounding_box_transform(np.array(part_bounding_box),
                                                 np.array(ref_part_bounding_box))
    vertices = part.get_vertices()
    test_orig = np.array([vertices])
    new_v = cv2.transform(test_orig, t)

    return [tuple(x) for x in new_v[0]]


def reconstruct_obj(part_list):
    # Create the output directory if it doesn't exist
    directory = "tmp_mixer_output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open("tmp_mixer_output/output_mesh.obj", "w")

    vertex_count = 0
    for num, (key, part) in enumerate(part_list.items()):
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
    # random.seed(80)
    # random.seed(346)  # change data[chosenParts[1]]["chair_back"][0] on line 131 to
    #                       data[chosenParts[1]]["chair_back"][3]
    index = random.choices(range(len(sub_folders)), k=4)

    chosenParts = [sub_folders[part] for part in index]

    print("Chosen parts are from folder:")
    print(chosenParts)

    parts_list = {}

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
                    parts_list[partName] = Part(obj["objs"], obj_Files)

                if partName == 'chair_back' and i == 1:
                    parts_list[partName] = Part(obj["objs"], obj_Files)

                if partName == 'chair_base' and i == 2:
                    parts_list[partName] = Part(obj["objs"], obj_Files)

                # TODO: Need to split arms into two data objects
                if partName == 'chair_arm' and i == 3:
                    parts_list[partName] = Part(obj["objs"], obj_Files)

    # Get the bounding_box data
    with open('mixer_files/bounding_box_data.json') as f:
        data = json.load(f)

    print('------Number of newly grabbed chair_back parts----')
    print(len(data[chosenParts[1]]["chair_back"]))

    # General TODO: parts are further divided into subparts, may further decide how to handle (ex. seed 346)
    # Transform chair back
    new_vertices = update_part_vertices(parts_list["chair_back"],
                                        data[chosenParts[1]]["chair_back"][0],
                                        data[chosenParts[0]]["chair_back"][0])
    parts_list["chair_back"].vertices = new_vertices

    print('------Number of newly grabbed chair_base parts----')
    print(len(data[chosenParts[2]]["chair_base"]))
    # Transform chair base
    new_vertices = update_part_vertices(parts_list["chair_base"],
                                        data[chosenParts[2]]["chair_base"][0],
                                        data[chosenParts[0]]["chair_base"][0])
    parts_list["chair_base"].vertices = new_vertices

    print('------Number of newly grabbed chair_arm parts----')
    print(len(data[chosenParts[3]]["chair_base"]))
    # TODO: May need additional handling due to some chairs not having arms (how to handle, we currently ignore)
    #  // two arms are regarded as separate two objects
    #
    # Transform chair arm
    # If reference chair has arms
    if len(data[chosenParts[0]]["chair_arm"]) and len(data[chosenParts[3]]["chair_arm"]):
        new_vertices = update_part_vertices(parts_list["chair_arm"],
                                            data[chosenParts[3]]["chair_arm"][0],
                                            data[chosenParts[0]]["chair_arm"][0])
        parts_list["chair_arm"].vertices = new_vertices

    # Combine the various part vertices/faces and then output obj files
    reconstruct_obj(parts_list)

