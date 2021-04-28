import struct
import moderngl
import moderngl_window as mglw
import numpy as np
import pyrr
import scipy.io
#import sys
import os

import PIL


def xRotationMatrix(theta):
    theta = theta*np.pi/180
    mat = pyrr.matrix44.create_identity()

    mat[1][1] = np.cos(theta)
    mat[1][2] = -np.sin(theta)
    mat[2][1] = np.sin(theta)
    mat[2][2] = np.cos(theta)

    return mat

def yRotationMatrix(theta):
    theta = theta*np.pi/180
    mat = pyrr.matrix44.create_identity()

    mat[0][0] = np.cos(theta)
    mat[0][2] = np.sin(theta)
    mat[2][0] = -np.sin(theta)
    mat[2][2] = np.cos(theta)

    return mat

def zRotationMatrix(theta):
    theta = theta*np.pi/180
    mat = pyrr.matrix44.create_identity()

    mat[0][0] = np.cos(theta)
    mat[0][1] = -np.sin(theta)
    mat[1][0] = np.sin(theta)
    mat[1][1] = np.cos(theta)

    return mat

def rotateMatrix(rotations):
    return np.matmul(np.matmul(xRotationMatrix(rotations[0]),yRotationMatrix(rotations[1])),zRotationMatrix(rotations[2]))

def translateMatrix(translations):
    mat = pyrr.matrix44.create_identity()
    mat[3][0] = translations[0]
    mat[3][1] = translations[1]
    mat[3][2] = translations[2]

    return mat

#Default to a unit cube
defaultVertexPositions = np.array([
    #// Front face
    -1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,

    #// Back face
    -1.0, -1.0, -1.0,
    -1.0,  1.0, -1.0,
     1.0,  1.0, -1.0,
     1.0, -1.0, -1.0,

    #// Top face
    -1.0,  1.0, -1.0,
    -1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0, -1.0,

    #// Bottom face
    -1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0,  1.0,
    -1.0, -1.0,  1.0,

    #// Right face
     1.0, -1.0, -1.0,
     1.0,  1.0, -1.0,
     1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,

    #// Left face
    -1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,
  ], dtype='f4')

defaultFaceIndices = np.array([
    0,  1,  2,      0,  2,  3,    # front
    4,  5,  6,      4,  6,  7,    # back
    8,  9,  10,     8,  10, 11,   # top
    12, 13, 14,     12, 14, 15,   # bottom
    16, 17, 18,     16, 18, 19,   # right
    20, 21, 22,     20, 22, 23,   # left
],dtype='i4')

vertexShaderSource = '''
    in vec4 aVertexPosition;
    //in vec4 aVertexColor;

    uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;

    //varying vec4 vColor;

    void main(void) {
        gl_Position = uProjectionMatrix * uModelViewMatrix * aVertexPosition;

        //gl_Position = uModelViewMatrix * aVertexPosition;
        //gl_Position = aVertexPosition;
        //vColor = aVertexColor;
    }
    '''

fragmentShaderSource = '''
    void main() {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    '''

flatten = lambda lis: [value for sub in lis for value in sub]

class Renderer:
    def __init__(self):
        self.context = moderngl.create_standalone_context()

        #Shaders
        self.vShaderSource = vertexShaderSource
        self.fShaderSource = fragmentShaderSource

        #Compiled shader program
        self.program = self.context.program(vertex_shader=self.vShaderSource, fragment_shader=self.fShaderSource)

        #Projection matrix
        self.projectionMatrix = pyrr.objects.matrix44.Matrix44.perspective_projection(90, 1, 0.01, -10, dtype='f4')    
        
        #Set uniforms
        self.program.__setitem__('uProjectionMatrix', tuple(flatten(self.projectionMatrix)))
        self.setView([0,0,0],[0,0,-2])

        #Create buffers and store default values
        self.setBuffers(defaultVertexPositions,defaultFaceIndices)

    #render a PIL image    
    def renderImage(self):
        #Set vertex array object to be rendered 
        self.vertexArray = self.context.vertex_array(self.program, [(self.vertexBuffer, "3f /v", "aVertexPosition" )], self.indexBuffer)

        #Clear the context
        self.context.clear(1.0, 1.0, 1.0)

        #Create a framebuffer for the render
        self.frameBuffer = self.context.simple_framebuffer((512, 512))

        #Use the new frame buffer
        self.frameBuffer.use()

        #Clear the new frame buffer
        self.frameBuffer.clear(0.0,0.0,0.0,1.0)
    
        #Render the image
        self.vertexArray.render(moderngl.TRIANGLES)

        #Convert the framebuffer to a PIL image
        return PIL.Image.frombytes('RGB', self.frameBuffer.size, self.frameBuffer.read(), 'raw', 'RGB', 0, -1)


    #set vertex buffer and index buffer
    def setBuffers(self, vertices, indices):
        self.vertexBuffer = self.context.buffer(vertices)
        if(indices.size == 0):
            indices = np.array([1,1,1],dtype='i4')
        self.indexBuffer = self.context.buffer(indices)
        return

    #set view uniform
    def setView(self, rotation, translation):
        '''
            Set the view matrices with the given rotation and translation values.
            
            Args:
                rotation (list): A list of [x, y, z] rotation values in degrees
                translation (list): A list of [x, y, z] translation values

            Returns:
                Nothing
        '''
        self.viewMatrix = np.matmul(translateMatrix(translation),rotateMatrix(rotation))
        self.program.__setitem__('uModelViewMatrix', tuple(flatten(self.viewMatrix)))
        return


    def getPart(self, directory, chairNumber, componentNumber, translate=True):

        if( translate and (chairNumber < 1 or chairNumber > 6201)):
            print("Error, chair number does not exist")
            return

        if( componentNumber < 0 or componentNumber > 3):
            print("Error, component number does not exist")
            return

        
        #get the obj and obb files
        if translate:

            #expects the grass-master directory
            pmiMat = scipy.io.loadmat(directory+"/Chair/part mesh indices/"+ str(chairNumber) +".mat")


            with open(directory+"/Chair/obbs/"+str(pmiMat['shapename'][0]+".obb"),"r") as obbFile:

                line = obbFile.readline()

                #get the labels from the obb
                while(line != '' and line[0] != 'L'):
                    line = obbFile.readline()

                line = obbFile.readline()

                labels = []

                while(line != ''):
                    labels.append(line.split()[0])
                    line = obbFile.readline()

                #Parse the object file
                with open(directory+"/Chair/models/"+str(pmiMat['shapename'][0]+".obj"),"r") as objFile:
                    lines = objFile.read().splitlines()
                    #print("Loading chair model: ",pmiMat['shapename'])

        else:
            with open(directory+"/Chair/obbs/"+str(chairNumber)+".obb","r") as obbFile:

                line = obbFile.readline()

                #get the labels from the obb
                while(line != '' and line[0] != 'L'):
                    line = obbFile.readline()

                line = obbFile.readline()

                labels = []

                while(line != ''):
                    labels.append(line.split()[0])
                    line = obbFile.readline()

                #Parse the object file
                with open(directory+"/Chair/models/"+str(chairNumber)+".obj","r") as objFile:
                    lines = objFile.read().splitlines()
                    #print("Loading chair model: ",chairNumber)

                
        


        
        
        groups = []
        vertices = []

        for line in lines:
            if line[0] == 'g':
                group = []
                groups.append(group)

            elif line[0] == 'v':
                vertices.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
                
            elif line[0] == 'f':
                group.append([int(line.split()[1])-1, int(line.split()[2])-1, int(line.split()[3])-1])

        faces = []
        for group in groups:
            if labels[groups.index(group)] == str(componentNumber):
                faces.append(group)

        self.setBuffers(np.array(vertices, dtype='f4'), np.array(flatten(faces), dtype='i4'))

        return


################################################################################################


#Need a function which loads part or parts of a single object file and loads them as a valid image
#Just need all vertices and relevant faces?

def threeAngleRender(renderer):

    angles = []

    renderer.setView([0,0,0],[0,0,-2])
    angles.append(renderer.renderImage())

    renderer.setView([-90,0,0],[0,-2,0])
    angles.append(renderer.renderImage())

    renderer.setView([0,-90,0],[2,0,0])
    angles.append(renderer.renderImage())
    
    return angles



"""def pixelCount(image):
    return np.sum(image)/255/3"""

#sum = PIL.Image.fromarray(test1+test2)

def intersect(image1, image2):
    return np.minimum(np.asarray(image1), np.asarray(image2))

def union(image1, image2):
    return np.maximum(np.asarray(image1), np.asarray(image2))

def intersectOverUnion(image1, image2):
    intersectValue = np.sum(intersect(image1,image2))
    unionValue = np.sum(union(image1, image2))
    if(unionValue == 0.0): return 1.0
    
    return intersectValue/unionValue


def calculateIOU(chairs, directory="../grass-master", translate=True):
    renderer = Renderer()

    masterRenders = []

    for i in range(4):
        renderer.getPart(directory, chairs[0], i, translate)
        masterRenders.append(threeAngleRender(renderer))

    partChairs = []
    for chair in range(1, len(chairs)):
        partChairs.append([])
        for i in range(4):
            renderer.getPart(directory, chairs[chair], i, translate)
            partChairs[chair-1].append(threeAngleRender(renderer))
    
    ious = []
    for part in range(4):
        ious.append([])
        for chair in range(len(partChairs)):
            ious[part].append((intersectOverUnion(masterRenders[part][0], partChairs[chair][part][0]) + intersectOverUnion(masterRenders[part][1], partChairs[chair][part][1]) + intersectOverUnion(masterRenders[part][2], partChairs[chair][part][2]))/3)

    return ious

#master = calculateIOU([369,175,5540])

#master = calculateIOU(['2585','2323','43872'], translate=False)

#print("done")
