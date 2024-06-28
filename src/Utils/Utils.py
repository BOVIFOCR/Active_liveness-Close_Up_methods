import cv2
from imutils import face_utils
from math import sqrt
import dlib

class Utils:

    show = lambda m,v: print(m) if v else None

    @staticmethod
    def printEpoch(prefix,epoch,loss,acc,hter,f1,verbose):
        
        tableHeader = ""
        for col in ["Epoch","Loss","Accuracy","HTER","F1"]:
            tableHeader = tableHeader + f"{prefix} {col}".ljust(15," ") + "|"
        tableContent = ""
        for val,col in zip([epoch,loss,acc,hter,f1],["Epoch","Loss","Accuracy","HTER","F1"]):
            
            if col in ["Accuracy","HTER"]:
                tableContent = tableContent + f"{(val):.3f}%".ljust(15," ") + ","
            elif col in ["Epoch"]:
                tableContent = tableContent + f"{int(val)}".ljust(15," ") + ","
            else:
                tableContent = tableContent + f"{(val):.4f}".ljust(15," ") + ","

        Utils.show(tableHeader,verbose)
        Utils.show(tableContent,verbose)
        return


    @staticmethod
    def loadFaceDetector(modelPath,verbose=False):
        faceDetector = None
        Utils.show("Loading face detector...",verbose)
        faceDetector = dlib.cnn_face_detection_model_v1(f"{modelPath}/mmod_human_face_detector.dat")
        Utils.show("Loaded!",verbose)
        return faceDetector
    
    
    @staticmethod
    def loadLandmarkExtractor(modelPath,verbose=False):
        landmarkExtractor = None
        Utils.show('Loading landmark Extractor...',verbose)
        landmarkExtractor = dlib.shape_predictor(f"{modelPath}/shape_predictor_68_face_landmarks.dat")
        Utils.show('Loaded!',verbose)
        return landmarkExtractor
    

    @staticmethod
    def convertAndTrimBB(image, rect):
        # extract the starting and ending (x, y)-coordinates of the
        # bounding box
        startX = rect.left()
        startY = rect.top()
        endX = rect.right()
        endY = rect.bottom()
        # ensure the bounding box coordinates fall within the spatial
        # dimensions of the image
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(endX, image.shape[1])
        endY = min(endY, image.shape[0])
        # compute the width and height of the bounding box
        w = endX - startX
        h = endY - startY
        # return our bounding box coordinates
        return (startX, startY, w, h)


    @staticmethod
    def detectMainFace(faceDetector,image):
        mainFace = None
        mainRect = None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(gray,0)

        if len(faces) == 0:
            # if no face was found
            mainFace = None
            mainRect = None
        else:
            # select the largest face
            mainFaceSize = 0
            for face in faces:
                rect = face.rect
                face = Utils.convertAndTrimBB(image,face.rect)
                faceSize = face[2] * face[3]
                if faceSize > mainFaceSize:
                    mainFace = face
                    mainRect = rect
                    mainFaceSize = faceSize
        return mainFace,mainRect
    

    @staticmethod
    def extractLandmarks(landmarkExtractor,image,roi):
        landmarks = None
        
        # convert to gray if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # extract landmarks
        landmarks = landmarkExtractor(image, roi)
        if landmarks is not None:
            landmarks = face_utils.shape_to_np(landmarks)
        
        return landmarks
    

    @staticmethod
    def distance(p0,p1):
        return sqrt((p0[0]-p1[0])*(p0[0]-p1[0]) + (p0[1]-p1[1])*(p0[1]-p1[1]))
    
    @staticmethod
    def normalizeVectors(vector0,vector1):
        if (vector0 is not None) and (vector1 is not None):
            return [v0/v1 for v0,v1 in zip(vector0,vector1)]
        else:
            return None
        
                