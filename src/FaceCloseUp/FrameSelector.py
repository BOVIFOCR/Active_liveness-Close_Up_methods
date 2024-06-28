import os
import cv2
import random
from src.Utils.Utils import Utils

class FrameSelector:

    @staticmethod
    # Try to get frame from given category
    # If fails, try to recursively get frame from category-1 and category+1
    def __pickFrameFromCategory(category,numCategories,frameCategory,selected,explored=[]):

        # If reached extreme categories
        if category == numCategories or category == -1 or category in explored:
            return None,explored,category
        
        explored.append(category)
        # Get frame from given category
        matched_category = [f for f,c in frameCategory.items() if c == category]
        available = [f for f in matched_category if not(f in selected)]
        # If does not have a frame from given category
        if len(available) == 0:
            # Try to recursively get frame from category-1 and category+1
            picked_lower,_,category_lower = FrameSelector.__pickFrameFromCategory(category-1,numCategories,frameCategory,selected,explored=explored)
            picked_upper,_,category_upper = FrameSelector.__pickFrameFromCategory(category+1,numCategories,frameCategory,selected,explored=explored)
            # Select frame from closest category
            if picked_lower is None and picked_upper is not None:
                return picked_upper,explored,category_upper
            elif picked_lower is not None and picked_upper is None:
                return picked_lower,explored,category_lower
            else:
                if abs(category_upper-category) == abs(category_lower-category):
                    if bool(random.getrandbits(1)):
                        return picked_lower,explored,category_lower
                    else:
                        return picked_upper,explored,category_upper    
                elif abs(category_upper-category) > abs(category_lower-category):
                    return picked_lower,explored,category_lower
                else:
                    return picked_upper,explored,category_upper
        else:
            return available[random.randint(0,len(available)-1)],explored,category


    @staticmethod
    # Generalized category classification
    def __categoryBySize(face,width,height,numCategories):
        # Face Close-up paper frame sizes
        originalSize = 1080*1920
        screenSize = width*height
        # Adapt to used dataset frame sizes
        conversionRatio = screenSize/originalSize
        # Category face size range
        stepSize = (0.85-0.15)/numCategories
        # New generalized range table
        rangeTable = {cat:[(0.15+cat*stepSize)*1000000*conversionRatio,(0.15+(cat+1)*stepSize)*1000000*conversionRatio] for cat in range(0,numCategories)}
        # Classify face
        if not(face is None):   
            faceSize = face[2]*face[3]
            for category in rangeTable:
                if faceSize > rangeTable[category][0] and faceSize <= rangeTable[category][1]:
                    return category

        return None

    def getFrameCategory(framePath,faceDetector,N):
        category = None
        frame = cv2.imread(framePath)
        height, width, _ = frame.shape
        # detect face
        face,_  = Utils.detectMainFace(faceDetector,frame)
        if face is not None:
            category = FrameSelector.__categoryBySize(face,width,height,N)
        return category


    @staticmethod
    def selectFrames(samplePath,N,faceDetector,landmarkExtractor,df_preProcessed,verbose):
        
        Utils.show(f"Selecting frames of sample {samplePath}...",verbose)
        refFrame = None
        pickedFrames = []
        frameCategory = {}

        videoFrames = os.listdir(samplePath)
        validFrames = []
        if df_preProcessed is None:
            # check valid frames
            for fileName in videoFrames:
                framePath = f"{samplePath}/{fileName}"
                category = FrameSelector.getFrameCategory(framePath,faceDetector,N)

                if category is not None:
                    frameCategory[fileName] = category
                    validFrames.append(fileName)
        else:
            # get pre-processed data
            validFrames = df_preProcessed["frame"].tolist()
            categories = df_preProcessed["category"].tolist()
            for fileName,category in zip(validFrames,categories):
                frameCategory[fileName] = category

        if len(validFrames) < (N+1):
            raise Exception(f"Not enough valid frames({len(validFrames)} of {len(videoFrames)}).")
        else:
            pickedFrames = []
            refFrame,_,_ = FrameSelector.__pickFrameFromCategory(0,N,frameCategory,pickedFrames,explored=[])

            for category in range(0,N):
                picked,_,_ = FrameSelector.__pickFrameFromCategory(category,N,frameCategory,pickedFrames+[refFrame],explored=[])
                if picked is None:
                    raise Exception(f"Could not pick a frame from category {category}.")
                pickedFrames.append(picked)

            if len(pickedFrames) != N:
                raise Exception(f"Erroneously picked {len(pickedFrames)} of {N} required.")
            else:
                Utils.show(f"Successifully picked {len(pickedFrames)} frames!",verbose)

        return [refFrame] + pickedFrames