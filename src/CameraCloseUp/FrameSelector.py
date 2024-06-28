import os
import cv2
from random import shuffle
from src.Utils.Utils import Utils

class FrameSelector:

    @staticmethod
    def __splitInBins(validFrames,nBins):
        
        # separate in bins
        idealBinSize = len(validFrames)//nBins
        remainer = len(validFrames) - nBins * idealBinSize
        binList = []
        currentBin = []
        for validFrame in validFrames:
            if remainer > 0:
                if len(currentBin) >= (idealBinSize + 1):
                    binList.append(currentBin)
                    currentBin = []
                    remainer -= 1
            else:
                if len(currentBin) >= idealBinSize:
                    binList.append(currentBin)
                    currentBin = []
            currentBin.append(validFrame)
        # append last bin
        if len(currentBin) > 0:
            binList.append(currentBin)

        return binList
    
    @staticmethod
    def __pickFromBin(bin,amount):
        shuffle(bin)
        return bin[:min(amount,len(bin))]

    @staticmethod
    def selectFrames(samplePath,N,bins,faceDetector,landmarkExtractor,df_preProcessed,verbose):
        
        Utils.show(f"Selecting frames of sample {samplePath}...",verbose)
        refFrame = None
        pickedFrames = []

        videoFrames = os.listdir(samplePath)
        validFrames = []
        if df_preProcessed is None:
            # check valid frames
            for fileName in videoFrames:
                # read frame
                frame = cv2.imread(f"{samplePath}/{fileName}")
                # detect face
                face,rect  = Utils.detectMainFace(faceDetector,frame)

                if face is not None:
                    # detect landmarks
                    landmarks = Utils.extractLandmarks(landmarkExtractor,frame,rect)
                    # if detected face and landmarks
                    # then it's a valid frame
                    if landmarks is not None:
                        validFrames.append(fileName)
        else:
            validFrames = df_preProcessed["frame"].tolist()

        if len(validFrames) < (N+1):
            raise Exception(f"Not enough valid frames({len(validFrames)} of {len(videoFrames)}).")
        else:
            Utils.show(f"{len(validFrames)} valid frames of {len(videoFrames)}",verbose)
            validFrames.sort()
            # pick reference frame
            refPosition = len(validFrames) // 2
            refFrame = validFrames.pop(refPosition)

            # separate in bins
            binList = FrameSelector.__splitInBins(validFrames,bins)
            
            idealSelection = N // bins
            remainer = N - bins * idealSelection
            for bin in binList:
                amount = idealSelection
                if remainer > 0:
                    remainer -= 1
                    amount += 1
                pickedFrames = pickedFrames + FrameSelector.__pickFromBin(bin,amount)

            if len(pickedFrames) != N:
                raise Exception(f"Erroneously picked {len(pickedFrames)} of {N} required.")
            else:
                Utils.show(f"Successifully picked {len(pickedFrames)} frames!",verbose)
                pickedFrames.sort()

        return [refFrame] + pickedFrames