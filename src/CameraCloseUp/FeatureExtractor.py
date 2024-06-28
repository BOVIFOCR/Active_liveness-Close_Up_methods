import cv2
import os
import pandas as pd
from src.Utils.Utils import Utils


class FeatureExtractor:

    @staticmethod
    def preProcess(samplePath,faceDetector,landmarkExtractor,device,verbose):
        Utils.show(f"\nPre-processing sample {samplePath}...",verbose)
        frames = os.listdir(samplePath)

        features = []
        signatureSize = None
        for fileName in frames:
            Utils.show(f"Extracting distortion features...",verbose)
            framePath = f"{samplePath}/{fileName}"
            signature = FeatureExtractor.__extractFrameSignature(framePath,faceDetector,landmarkExtractor)
            signatureSize = len(signature)
            features.append([fileName]+signature)
            
        df_features = None
        if signatureSize is not None:
            df_features = pd.DataFrame(features,columns=["frame"]+[f"distortion_feature_{i}" for i in range(0,signatureSize)])

        return df_features

    @staticmethod
    def __computeSignature(landmarks):
        distanceVector = []
        for i in range(0,len(landmarks)):
            for j in range(i+1,len(landmarks)):
                if i >= 48 and j >= 48:
                    # exclude distances between mouth landmarks 
                    continue
                dist = Utils.distance(landmarks[i],landmarks[j])
                if dist != 0:
                    distanceVector.append(dist)
                else:
                    distanceVector.append(1)
        return distanceVector
    
    
    def __extractFrameSignature(framePath,faceDetector,landmarkExtractor):
        signature = None
        frame = cv2.imread(framePath)
        # detect face and extract landmarks
        face,rect = Utils.detectMainFace(faceDetector,frame)
        if face is not None:
            landmarks = Utils.extractLandmarks(landmarkExtractor,frame,rect)
            if landmarks is not None:
                # compute signature
                signature = FeatureExtractor.__computeSignature(landmarks)

        if signature is None:
            raise Exception(f"Could not extract signature of frame {framePath}.")
        return signature


    def __getPreProcesseFeature(df_sample,featurePrefix,frame):
    
        allCols = df_sample.columns.tolist()
        featureCols = [s for s in allCols if s.startswith(featurePrefix)]
        signature = None
        # Check if frame was processed
        if len(df_sample.loc[df_sample["frame"]==frame]) > 0:
            # fetch frame row
            occurence = df_sample.loc[df_sample["frame"]==frame].iloc[0]
            signature = occurence[featureCols].tolist()
        return signature

    @staticmethod
    def getDistortionFeatures(samplePath,sample,N,faceDetector,landmarkExtractor,df_sample,verbose):
        Utils.show(f"Extracting distortion features of sample {samplePath}...",verbose)
        distortionFeatures = []

        refFramePath = f"{samplePath}/{sample['frame_ref']}"
        refSignature = None
        if df_sample is not None:
            # try to get pre processed feature vector
            refSignature = FeatureExtractor.__getPreProcesseFeature(df_sample,"distortion_feature_",sample['frame_ref'])
        if refSignature is None:
            # compute distortion feature vector
            refSignature = FeatureExtractor.__extractFrameSignature(refFramePath,faceDetector,landmarkExtractor)

        for i in range(0,N):
            # Extract signature of selected frame 
            framePath = f"{samplePath}/{sample[f'frame_{i}']}"
            frameSignature = None
            
            if df_sample is not None:
                # try to get pre processed feature vector
                frameSignature = FeatureExtractor.__getPreProcesseFeature(df_sample,"distortion_feature_",sample[f'frame_{i}'])
            if frameSignature is None:
                # compute distortion feature vector
                frameSignature = FeatureExtractor.__extractFrameSignature(framePath,faceDetector,landmarkExtractor)
            # Normalize using ref frame signature
            normalizedSignature = Utils.normalizeVectors(frameSignature,refSignature)
            # Store values
            distortionFeatures = distortionFeatures + normalizedSignature
    
        return distortionFeatures