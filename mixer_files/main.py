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


def split_chair_arm(bounding_boxes):
    left_arm = []
    right_arm = []
    for box in bounding_boxes:
        # positive x is left arm from sitting POV
        if box[0][0] > 0:
            left_arm.append(box)
        else:
            right_arm.append(box)

    return left_arm, right_arm


def update_part_vertices(part, part_bounding_box, ref_part_bounding_box):
    t = geometric_helpers.bounding_box_transform(np.array(part_bounding_box),
                                                 np.array(ref_part_bounding_box))
    vertices = part.get_vertices()
    test_orig = np.array([vertices])
    new_v = cv2.transform(test_orig, t)

    return [tuple(x) for x in new_v[0]]


def transform_part(part_obj, from_bounding_box, to_bounding_box):
    # There are issues with the affine transform of the bounding box are exactly the same (no transform required)
    if from_bounding_box != to_bounding_box:
        new_vertices = update_part_vertices(part_obj,
                                            from_bounding_box,
                                            to_bounding_box)
        part_obj.vertices = new_vertices


def reconstruct_obj(part_list, index):
    # Create the output directory if it doesn't exist
    directory = "tmp_mixer_output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open("tmp_mixer_output/output_mesh_" + str(index) + ".obj", "w")

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
    parser = argparse.ArgumentParser(description='Chairs to create, chairs directory')
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--dir", required=True)

    args = parser.parse_args()

    if not args.dir:
        print('Need to provide chair directory in run configurations')
        exit(-1)

    chairs_to_create = args.count;
    data_dir = args.dir
    sub_folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    for current_chair_index in range(chairs_to_create):
        #sub_folders = ['37107', '39781', '40141', '39426', '35698', '2320', '40546', '37790', '43006', '37108']
        index = random.choices(range(len(sub_folders)), k=4)
        chosenParts = [sub_folders[part] for part in index]

        # -------- TESTING ------------
        # chosenParts = ['172', sub_folders[index[1]], '173', '178']
        # issue with part '41542'
        # chosenParts = ['172', '41542', '173', '178']
        # chosenParts = ['44919', '43068', '36920', '42945']
        # chosenParts = ['38104', '39815', '39986', '43754']
        # chosenParts = ['40996', '3124', '43388', '44352']
        # chosenParts = ['39040', '41753', '42363', '43605']
        # -------- TESTING ------------

        print("Chosen parts are from folder:")
        print(chosenParts)

        parts_list = {}
        ref_parts_list = {}

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

                arm_count = 0
                ref_arm_count = 0

                for obj in mergedData[0]['children']:
                    partName = obj['name']
                    if i == 0:
                        if partName == 'chair_arm':
                            ref_arm_count = ref_arm_count + 1
                            ref_parts_list[partName + str(ref_arm_count)] = Part(obj["objs"], obj_Files, part_id)
                        else:
                            ref_parts_list[partName] = Part(obj["objs"], obj_Files, part_id)

                        if partName == 'chair_seat':
                            parts_list[partName] = Part(obj["objs"], obj_Files, part_id)

                    if partName == 'chair_back' and i == 1:
                        parts_list[partName] = Part(obj["objs"], obj_Files, part_id)
                    
                    if partName == 'chair_base' and i == 2:
                        parts_list[partName] = Part(obj['objs'], obj_Files, part_id)

                    if partName == 'chair_arm' and i == 3:
                        arm_count = arm_count + 1
                        parts_list[partName + str(arm_count)] = Part(obj["objs"], obj_Files, part_id)

        # Get the bounding_box data
        with open('mixer_files/bounding_box_data.json') as f:
            data = json.load(f)

            # Pre-parse chair arms if chair_arm exists
            # We need to parse the bounding box data as both chair arms are gathered within the same object
            if len(data[chosenParts[3]]['chair_arm']):
                left, right = split_chair_arm(data[chosenParts[3]]['chair_arm'])
                if len(left) > 1:
                    left = geometric_helpers.agg_boxes(left)

                parts_list['chair_arm1'].set_bounding_box(left)

                if len(right) > 1:
                    right = geometric_helpers.agg_boxes(right)

                parts_list['chair_arm2'].set_bounding_box(right)

            if len(data[chosenParts[0]]['chair_arm']):
                left, right = split_chair_arm(data[chosenParts[3]]['chair_arm'])
                if len(left) > 1:
                    left = geometric_helpers.agg_boxes(left)

                ref_parts_list['chair_arm1'].set_bounding_box(left)

                if len(right) > 1:
                    right = geometric_helpers.agg_boxes(right)

                ref_parts_list['chair_arm2'].set_bounding_box(right)

        # Set the bounding boxes for the reference and new parts
        ignore_list = ['chair_arm1', 'chair_arm2']
        for new_part in parts_list.keys():
            cur_part_id = parts_list[new_part].get_part_id()
            if new_part not in ignore_list:  # since they are pre-parsed above
                parts_list[new_part].set_bounding_box(geometric_helpers.agg_boxes(data[cur_part_id][new_part]))

        for ref_part in ref_parts_list.keys():
            cur_part_id = ref_parts_list[ref_part].get_part_id()
            if ref_part not in ignore_list:  # since they are pre-parsed above
                ref_parts_list[ref_part].set_bounding_box(geometric_helpers.agg_boxes(data[cur_part_id][ref_part]))

        ###############################################################################################
        print('------Number of newly grabbed chair_back parts----')
        print(len(data[chosenParts[1]]["chair_back"]))

        # Transform chair back
        chosen_back = parts_list["chair_back"].get_bounding_box()
        ref_back = ref_parts_list["chair_back"].get_bounding_box()
        transform_part(parts_list["chair_back"], chosen_back, ref_back)

        # # Get the seat of the new back to determine transformation
        # seat_part = data[parts_list["chair_back"].get_part_id()]["chair_seat"]
        # seat_bounding_box = geometric_helpers.agg_boxes(seat_part)
        #
        # # Check the y difference between new chair_back and its seat starting from the lowest point
        # new_part_diff = chosen_back[1][1] - seat_bounding_box[0][1]
        # ref_part_diff = ref_parts_list["chair_seat"].get_bounding_box()[0][1] - ref_back[1][1]
        # diff = new_part_diff - ref_part_diff
        #
        # print(diff)

        # Shift the new back by the difference to make sure the bottom of the back is aligned with the seat
        # parts_list["chair_back"].vertices = geometric_helpers.shift_vertices(parts_list["chair_back"], "chair_back", diff)

        #########################################################################################################
        print('------Number of newly grabbed chair_base parts----')
        print(len(data[chosenParts[2]]["chair_base"]))

        chosen_base = parts_list["chair_base"].get_bounding_box()       # Get the chair base's bounding box
        ref_seat = ref_parts_list["chair_seat"].get_bounding_box()      # Get the reference chair's seat bounding box
        
        if 'chair_base' in ref_parts_list.keys():
            ref_base = ref_parts_list["chair_base"].get_bounding_box()      # Get the reference chair's base bounding box
        else:
            ref_base = chosen_base
        
        seat_width_buffer = abs(ref_seat[0][0] - ref_seat[2][0]) / 10
        
        # Calculate new parts bounding box as a combination of the reference pieces and new piece
        maxx = ref_seat[0][0] - seat_width_buffer
        minx = ref_seat[2][0] + seat_width_buffer
        
        maxy = (ref_seat[0][1] + ref_seat[1][1]) / 2
        miny = maxy - (0.66*(chosen_base[0][1] - chosen_base[1][1]) + 0.33*(ref_base[0][1] - ref_base[1][1]))
        
        maxz = ref_seat[0][2]
        minz = ref_seat[4][2]
        
        new_box = [[maxx, maxy, maxz], [maxx, miny, maxz], [minx, maxy, maxz], [minx, miny, maxz],
                   [maxx, maxy, minz], [maxx, miny, minz], [minx, maxy, minz], [minx, miny, minz]]

        transform_part(parts_list["chair_base"], chosen_base, new_box)

        ######################################################################################################
        print('------Number of newly grabbed chair_arm parts----')
        print(len(data[chosenParts[3]]["chair_base"]))
        # TODO: May need additional handling due to some chairs not having arms (how to handle, we currently ignore)
        #  // two arms are regarded as separate two objects
        #
        # Transform chair arm
        # If reference chair has arms
        if len(data[chosenParts[0]]["chair_arm"]) and len(data[chosenParts[3]]["chair_arm"]):
            #Transform first arm
            chosen_arm1 = parts_list["chair_arm1"].get_bounding_box()
            ref_arm1 = ref_parts_list["chair_arm1"].get_bounding_box()
            transform_part(parts_list["chair_arm1"], chosen_arm1, ref_arm1)

            # transform second arm
            chosen_arm2 = parts_list["chair_arm2"].get_bounding_box()
            ref_arm2 = ref_parts_list["chair_arm2"].get_bounding_box()
            transform_part(parts_list["chair_arm2"], chosen_arm2, ref_arm2)

        # Combine the various part vertices/faces and then output obj files
        reconstruct_obj(parts_list, current_chair_index)
