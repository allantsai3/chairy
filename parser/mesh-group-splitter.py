
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
        
    labelCount = [0, 0, 0, 0]

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

        label = int(findLabels()[len(groupObjects)-1])
        labelList = ["back", "seat", "leg", "armrest"]
        labelCount[label] += 1
        

        #file name
        newFileName = "./splitFiles/" + os.path.basename(sys.argv[1]).split(".")[0] + "-" + str(len(groupObjects)) + labelList[label] + str(labelCount[label]) + ".obj"

        Path("./splitFiles").mkdir(parents=True, exist_ok=True)

        newGroupObj = open(newFileName, "w")
        newGroupObj.write(groupObjects[-1])
        newGroupObj.close()

        localFaceList = []
        localVertexList = []

#Finds the semantic labels for the segments of the given model file
def findLabels():

    readFile = open(os.path.dirname(os.path.dirname(sys.argv[1]))+'/obbs/'+os.path.basename(sys.argv[1]).split(".")[0]+'.obb', "r")

    line = readFile.readline()

    while(line != '' and line[0] != 'L'):
        line = readFile.readline()

    line = readFile.readline()

    labels = []

    while(line != ''):
        labels.append(line.split()[0])
        line = readFile.readline()

    return labels

if __name__ == "__main__":
    splitSegmentedMesh()

