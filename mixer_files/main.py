import os
import random
import json
import cv2
import argparse
from Part import Part
import numpy as np
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
        point_distance_error = calculate_error(original_part['bounding_box'], c['bounding_box'])
        
        if point_distance_error < current_pair[1]:
            current_pair = (c, point_distance_error)
        
    return current_pair[0]


def point_in_box(point, box):
    return (point[0] >= box[0][0] and point[0] >= box[1][0] and point[0] >= box[2][0] and point[0] >= box[3][0] and point[0] <= box[4][0] and point[0] <= box[5][0] and point[0] <= box[6][0] and point[0] <= box[7][0]) and (point[1] >= box[0][1] and point[1] >= box[1][1] and point[1] <= box[2][1] and point[1] <= box[3][1] and point[1] >= box[4][1] and point[1] >= box[5][1] and point[1] <= box[6][1] and point[1] <= box[7][1])  and (point[2] >= box[0][2] and point[2] <= box[1][2] and point[2] >= box[2][2] and point[2] <= box[3][2] and point[2] >= box[4][2] and point[2] <= box[5][2] and point[2] >= box[6][2] and point[2] <= box[7][2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chairs directory')
    parser.add_argument("--dir", required=True)

    args = parser.parse_args()

    if not args.dir:
        print('Need to provide chair directory in run configurations')
        exit(-1)

    data_dir = args.dir

    sub_folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # -------- TESTING ------------
    # random.seed(23)
    # random.seed(80)
    # random.seed(346)  # change data[chosenParts[1]]["chair_back"][0] on line 131 to
    #                       data[chosenParts[1]]["chair_back"][3]
    #
    # -------- TESTING ------------
    index = random.choices(range(len(sub_folders)), k=4)
    chosenParts = [sub_folders[part] for part in index]

    # -------- TESTING ------------
    # chosenParts = ['172', sub_folders[index[1]], '173', '178']
    # issue with part '41542'
    # chosenParts = ['172', '41542', '173', '178']
    # chosenParts = ['35824', '43115', '41157', '42831']
    # -------- TESTING ------------

    print("Chosen parts are from folder:")
    print(chosenParts)

    parts_list = {}
    ref_parts_list = {}

    """
    {
        original_base: json_data
        original_base_path: path
        replacement_base: json_data
        replacement_base_path: path
    }
    """
    chair_bases = {}

    # Choose the chair_seat, chair_back, chair_base, chair_arm
    for i, part_id in enumerate(chosenParts):
        with open(os.path.join(data_dir, part_id, "result_after_merging.json")) as json_file:
            try:
                mergedData = json.load(json_file)
            except ValueError:
                print(part_id)
            except IOError:
                print('IOError')

            obj_Files = os.path.join(data_dir, part_id, "objs")

            for obj in mergedData[0]['children']:
                partName = obj['name']
                if i == 0:
                    ref_parts_list[partName] = Part(obj["objs"], obj_Files, part_id)
                    if partName == 'chair_seat':
                        parts_list[partName] = Part(obj["objs"], obj_Files, part_id)

                if partName == 'chair_back' and i == 1:
                    parts_list[partName] = Part(obj["objs"], obj_Files, part_id)

                if partName == 'chair_base':
                    if i == 0:
                        # Here I am going to keep a reference to the original chairs parts
                        chair_bases['original_base'] = obj['children']
                        chair_bases['original_base_path'] = obj_Files
                    elif i == 2:
                        #parts_list[partName] = Part(obj["objs"], obj_Files)
                        # Here I am saving the parts that will replace the original chair parts
                        chair_bases['replacement_base'] = obj['children']
                        chair_bases['replacement_base_path'] = obj_Files

                # TODO: Need to split arms into two data objects
                if partName == 'chair_arm' and i == 3:
                    parts_list[partName] = Part(obj["objs"], obj_Files, part_id)

    # Get the bounding_box data
    with open('mixer_files/bounding_box_data.json') as f:
        data = json.load(f)

        # Set the bounding boxes for the reference and new parts
        for new_part in parts_list.keys():
            cur_part_id = parts_list[new_part].get_part_id()
            parts_list[new_part].set_bounding_box(geometric_helpers.agg_boxes(data[cur_part_id][new_part]))

        for ref_part in ref_parts_list.keys():
            cur_part_id = ref_parts_list[ref_part].get_part_id()
            ref_parts_list[ref_part].set_bounding_box(geometric_helpers.agg_boxes(data[cur_part_id][ref_part]))

    print('------Number of newly grabbed chair_back parts----')
    print(len(data[chosenParts[1]]["chair_back"]))

    print('----Number of reference chair_back parts----')
    print(len(data[chosenParts[0]]["chair_back"]))

    # Option 1: Extend the bounding box for both the reference and newagg_boxes(data[chosenParts[0]]["chair_back"])
    # NOT DONE: Option 2: Randomly choose symmetric parts from a list to replace

    # geometric_helpers.plot_bounding_box(geometric_helpers.agg_boxes(data[chosenParts[0]]["chair_back"]), geometric_helpers.agg_boxes(data[chosenParts[0]]["chair_seat"]), geometric_helpers.agg_boxes(data[chosenParts[1]]["chair_back"]))

    # General TODO: parts are further divided into subparts, may further decide how to handle (ex. seed 346)
    # Transform chair back
    chosen_back = parts_list["chair_back"].get_bounding_box()
    ref_back = ref_parts_list["chair_back"].get_bounding_box()
    new_vertices = update_part_vertices(parts_list["chair_back"],
                                        chosen_back,
                                        ref_back)
    parts_list["chair_back"].vertices = new_vertices

    # Check the bounding box of the new back's seat
    seat_part = data[parts_list["chair_back"].get_part_id()]["chair_seat"]
    seat_bounding_box = geometric_helpers.agg_boxes(seat_part)

    # 4 closest point are index 0, 2, 4, 6 (A, C, E, G)
    # Check the y difference between new chair_back and its seat
    new_part_diff = chosen_back[1][1] - seat_bounding_box[0][1]
    ref_part_diff = ref_back[1][1] - ref_parts_list["chair_seat"].get_bounding_box()[0][1]
    diff = new_part_diff - ref_part_diff

    # Shift the new back by the difference to make sure the bottom of the back is aligned with the seat
    parts_list["chair_back"].vertices = geometric_helpers.shift_vertices(parts_list["chair_back"], "chair_back", diff)

    print('------Number of newly grabbed chair_base parts----')
    print(len(data[chosenParts[2]]["chair_base"]))

    original_base_data = []  # Using these lists to collect data on the chair parts from the original and replacement models
    replacement_base_data = []
    for i, og_part in enumerate(chair_bases['original_base'][0]['children']):   # Enumerating through the children of the original chairs base (So the components of the chair base)
        part_data = {}
        part_data['name'] = og_part['name']
        part_data['objs'] = og_part['objs']
        part_data['bounding_box'] = data[chosenParts[0]]["chair_base"][i] # Doesn't work this way? The bounding boxes might not be in the same order?
        part_data['path'] = chair_bases['original_base_path']

        original_base_data.append(part_data)
    
    for j, rp_part in enumerate(chair_bases['replacement_base'][0]['children']):
        part_data = {}
        part_data['name'] = rp_part['name']
        part_data['objs'] = rp_part['objs']
        part_data['bounding_box'] = data[chosenParts[2]]["chair_base"][j]
        part_data['path'] = chair_bases['replacement_base_path']
        
        replacement_base_data.append(part_data)
 
    # For each component of the original chair base, find something to replace it with
    for i, original_part in enumerate(original_base_data):
        replacement_part = find_replacement(original_part, replacement_base_data) # Picks the part with the smallest error metric
        new_name = original_part['name'] + " " + str(i) # Just coming up with a generic name here
        parts_list[new_name] = Part(replacement_part["objs"], replacement_part['path']) # Create a new part object
        new_verts = update_part_vertices(parts_list[new_name], replacement_part['bounding_box'], original_part['bounding_box']) # Apply the transformation
        parts_list[new_name].vertices = new_verts

    """
    # Transform chair base
    new_vertices = update_part_vertices(parts_list["chair_base"],
                                        data[chosenParts[2]]["chair_base"][0],
                                        data[chosenParts[0]]["chair_base"][0])
    parts_list["chair_base"].vertices = new_vertices
    """
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

