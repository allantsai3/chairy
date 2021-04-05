
import sys
import os
#import functools
from pathlib import Path

#Takes the file name of the obj file to be split as an argument
#Writes each group as an independent OBJ file in the ./splitFiles directory

def splitSegmentedMesh(): 

    #print( 'Arguments: ',str(sys.argv) )
    if ( len(sys.argv) > 1 ):
        modelFile = open(sys.argv[1])

    else: 
        print("Please give a file name as an argument")
        return
        

    vertexList = []
    groupObjects = []

    localVertexList = []
    localFaceList = []

    modelLine = modelFile.readline()
    
    while (modelLine != ''):
        #For each group

        #verify another group is coming
        if modelLine == 'g %d\n' % len(groupObjects ):

            #Add a new object to the list
            groupObjects.append("")

            modelLine = modelFile.readline()
            continue

        
        #Add the vertices to the global list
        while(modelLine[0] == 'v'):
            vertexList.append(modelLine)
            modelLine = modelFile.readline()

        #Add the faces to the object's list, as well as any relevant vertices
        while(modelLine != '' and modelLine[0] == 'f'):
            localFaceList.append(modelLine)

            for x in modelLine.split():
                if x not in localVertexList:
                    localVertexList.append(x)

            modelLine = modelFile.readline()
        localVertexList.remove('f')


        #Create a list of vertices for the segment
        for x in localVertexList:
            groupObjects[-1] += vertexList[int(x)-1]

        #Replicates a given face with new vertex indices
        def newVertexIndex(strInput):
            #extract the vertices in the face
            verts = strInput.split()[1:]

            ret = map(lambda x : str(localVertexList.index(x)+1) , verts)

            return "f " + " ".join(ret) + "\n"

        #apply new vertex indices to the faces
        localFaceList = list(map(newVertexIndex, localFaceList))
    
        for x in localFaceList:
            groupObjects[-1] += x

        #print("finished group: ", len(groupObjects))

        #file name
        newFileName = "./splitFiles/" + os.path.basename(sys.argv[1]).split(".")[0] + "-" + str(len(groupObjects)) + ".obj"

        Path("./splitFiles").mkdir(parents=True, exist_ok=True)

        newGroupObj = open(newFileName, "w")
        newGroupObj.write(groupObjects[-1])
        newGroupObj.close()

        localFaceList = []
        localVertexList = []

    #grassdata = GRASSDataset('chair',3)
    #for i in range(len(grassdata)):
    #    boxes = decode_structure(grassdata[i].root)
    #    showGenshape(boxes)


# def safeIndex(listIn, val):
#     try:
#         index_val = listIn.index(val)
#     except ValueError:
#         index_val = -1
#     return index_val



if __name__ == "__main__":
    splitSegmentedMesh()

