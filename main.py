import os
import random
import json
import cv2
import argparse
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

def calculate_error(box1, box2):
    shift = (box1[0][0] - box2[0][0], box1[0][1] - box2[0][1], box1[0][2] - box2[0][2]) # Add this value to each box2 point
    error = 0
    for p1 in box1:
        min_distance = float('inf')
        for p2 in box2:
            d = (p1[0] - (p2[0] + shift[0]))**2 + (p1[1] - (p2[1] + shift[1]))**2 + (p1[2] - (p2[2] + shift[2]))**2
            min_distance = min(min_distance, d)
        error = error + min_distance
    return error

def find_replacement(original_part, replacement_base_data):
    # Grab ones with the same labels
    candidates = [part for part in replacement_base_data if part['name'] == original_part['name']]

    if len(candidates) == 0:    # If we don't get any hits then we're gonna take the best we can
        candidates = replacement_base_data
    
    current_pair = (None, float('inf'))
    for c in candidates:
        point_distance_error = calculate_error(original_part['bounding_box_1'], c['bounding_box_1'])
        
        if point_distance_error < current_pair[1]:
            current_pair = (c, point_distance_error)
        
    return current_pair[0]
    
def point_in_box(point, box):
    return (point[0] >= box[0][0] and point[0] >= box[1][0] and point[0] >= box[2][0] and point[0] >= box[3][0] and point[0] <= box[4][0] and point[0] <= box[5][0] and point[0] <= box[6][0] and point[0] <= box[7][0]) and (point[1] >= box[0][1] and point[1] >= box[1][1] and point[1] <= box[2][1] and point[1] <= box[3][1] and point[1] >= box[4][1] and point[1] >= box[5][1] and point[1] <= box[6][1] and point[1] <= box[7][1])  and (point[2] >= box[0][2] and point[2] <= box[1][2] and point[2] >= box[2][2] and point[2] <= box[3][2] and point[2] >= box[4][2] and point[2] <= box[5][2] and point[2] >= box[6][2] and point[2] <= box[7][2])
    
def create_bounding_box(part):
    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')
    min_z = float('inf')
    max_z = -float('inf')
    
    for v in part.vertices:
        min_x = min(min_x, v[0])
        max_x = max(max_x, v[0])
        
        min_y = min(min_y, v[1])
        max_y = max(max_y, v[1])
        
        min_z = min(min_z, v[2])
        max_z = max(max_z, v[2])

    return [[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z], [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z], [max_x, max_y, min_z], [max_x, max_y, max_z]]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chairs directory')
    parser.add_argument("--dir", required=True)

    args = parser.parse_args()

    if not args.dir:
        print('Need to provide chair directory in run configurations')
        exit(-1)

    data_dir = args.dir

    sub_folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # random.seed(23)
    # random.seed(80)
    # random.seed(346)  # change data[chosenParts[1]]["chair_back"][0] on line 131 to
    #                       data[chosenParts[1]]["chair_back"][3]
    index = random.choices(range(len(sub_folders)), k=4)

    #chosenParts = [sub_folders[part] for part in index]
    #chosenParts = ['172', '173', '176', '178']
    chosenParts = ['45054', '44204', '2558', '40584']

    print("Chosen parts are from folder:")
    print(chosenParts)

    parts_list = {}
    chair_bases = {}

    # Choose the chair_seat, chair_back, chair_base, chair_arm
    for i, part in enumerate(chosenParts):
        with open(os.path.join(data_dir, part, "result_after_merging.json")) as json_file:
            try:
                mergedData = json.load(json_file)
            except ValueError:
                print(part)
            except IOError:
                print('IOError')

            obj_Files = os.path.join(data_dir, part, "objs")

            for obj in mergedData[0]['children']:
                partName = obj['name']
                if partName == 'chair_seat' and i == 0:
                    parts_list[partName] = Part(obj["objs"], obj_Files)

                if partName == 'chair_back' and i == 1:
                    parts_list[partName] = Part(obj["objs"], obj_Files)

                if partName == 'chair_base':
                    if i == 0:
                        # Here I am going to keep a reference to the original chairs parts
                        
                        if 'children' in obj:
                            chair_bases['original_base'] = obj['children'][0]
                            chair_bases['original_pieces'] = True
                        else:                            
                            chair_bases['original_base'] = obj
                            chair_bases['original_pieces'] = False
                        
                        chair_bases['original_base_path'] = obj_Files
                    elif i == 2:
                        # Here I am saving the parts that will replace the original chair parts
                        if 'children' in obj:
                            chair_bases['replacement_base'] = obj['children'][0]
                            chair_bases['replacement_pieces'] = True
                        else:                            
                            chair_bases['replacement_base'] = obj
                            chair_bases['replacement_pieces'] = False
                        
                        chair_bases['replacement_base_path'] = obj_Files

                # TODO: Need to split arms into two data objects
                if partName == 'chair_arm' and i == 3:
                    parts_list[partName] = Part(obj["objs"], obj_Files)

    # Get the bounding_box data
    with open('bounding_box_data.json') as f:
        data = json.load(f)

    print('------Number of newly grabbed chair_back parts----')
    print(len(data[chosenParts[1]]["chair_back"]))

    # General TODO: parts are further divided into subparts, may further decide how to handle (ex. seed 346)
    # Transform chair back
    new_vertices = update_part_vertices(parts_list["chair_back"],
                                        data[chosenParts[1]]["chair_back"][0],
                                        data[chosenParts[0]]["chair_back"][0])
    parts_list["chair_back"].vertices = new_vertices


    ###
    print('------Number of newly grabbed chair_base parts----')
    print(len(data[chosenParts[2]]["chair_base"]))
	

    original_base_data = []     # Using these lists to collect data on the chair parts from the original and replacement models
    replacement_base_data = []
    
    if chair_bases['original_pieces']:
        for i, og_part in enumerate(chair_bases['original_base']['children']):   # Enumerating through the children of the original chairs base (So the components of the chair base)
            part_data = {}
            part_data['name'] = og_part['name']
            part_data['objs'] = og_part['objs']
            part_data['bounding_box_1'] = data[chosenParts[0]]["chair_base"][i] # Doesn't work this way? The bounding boxes might not be in the same order?
            part_data['path'] = chair_bases['original_base_path']
            part_data['bounding_box_2'] = create_bounding_box(Part(part_data['objs'], part_data['path']))
            original_base_data.append(part_data)
    else:
        part_data = {}
        part_data['name'] = chair_bases['original_base']['name']
        part_data['objs'] = chair_bases['original_base']['objs']
        part_data['bounding_box_1'] = data[chosenParts[0]]["chair_base"][0]
        part_data['path'] = chair_bases['original_base_path']
        part_data['bounding_box_2'] = create_bounding_box(Part(part_data['objs'], part_data['path']))
        original_base_data.append(part_data)
        
    if chair_bases['replacement_pieces']:    
        
        print(len(chair_bases['replacement_base']['children']))
        
        for j, rp_part in enumerate(chair_bases['replacement_base']['children']):
            part_data = {}
            part_data['name'] = rp_part['name']
            part_data['objs'] = rp_part['objs']
            part_data['bounding_box_1'] = data[chosenParts[2]]["chair_base"][j]
            part_data['path'] = chair_bases['replacement_base_path']
            part_data['bounding_box_2'] = create_bounding_box(Part(part_data['objs'], part_data['path']))

            replacement_base_data.append(part_data)
    else:
        part_data = {}
        part_data['name'] = chair_bases['replacement_base']['name']
        part_data['objs'] = chair_bases['replacement_base']['objs']
        part_data['bounding_box_1'] = data[chosenParts[2]]["chair_base"][0]
        part_data['path'] = chair_bases['original_base_path']
        part_data['bounding_box_2'] = create_bounding_box(Part(part_data['objs'], part_data['path']))
        replacement_base_data.append(part_data)
 
    # For each component of the original chair base, find something to replace it with
    for i, original_part in enumerate(original_base_data):
        replacement_part = find_replacement(original_part, replacement_base_data) # Picks the part with the smallest error metric
        new_name = original_part['name'] + " " + str(i) # Just coming up with a generic name here
        
        parts_list[new_name] = Part(replacement_part["objs"], replacement_part['path']) # Create a new part object
    
        new_verts = update_part_vertices(parts_list[new_name], replacement_part['bounding_box_2'], original_part['bounding_box_1']) # Apply the transformation
        
        parts_list[new_name].vertices = new_verts

    ###
	
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

