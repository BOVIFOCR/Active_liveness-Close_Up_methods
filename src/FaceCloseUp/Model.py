import pandas as pd
import torch.nn as torchNN
import torch
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
from src.Utils.Utils import Utils
from src.Utils.train_model import run_one_epoch
from src.FaceCloseUp.FrameSelector import FrameSelector
from src.FaceCloseUp.FeatureExtractor import FeatureExtractor
from src.FaceCloseUp.Classifier import Classifier
from src.FaceCloseUp.CloseUpLivenessDataset import CloseUpLivenessDataset

class Model:

    @staticmethod
    def preProcessDataset(df,source,N,faceDetector,landmarkExtractor,output,verbose=False):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df_live = df.loc[df["label"]=="live"]
        df_spoof = df.loc[df["label"]=="spoof"]
        # pre-process live and spoof samples
        for df,label in zip([df_live,df_spoof],["live","spoof"]):
            df_samples = None
            for _,row in df.iterrows():
                try:
                    samplePath = f"{source}/{label}/{row['sampleID']}"
                    preProcessedSample = FeatureExtractor.preProcess(N,samplePath,faceDetector,landmarkExtractor,verbose)
                    preProcessedSample["sampleID"] = row['sampleID']
                    preProcessedSample["label"] = label
                    
                    if df_samples is None:
                        df_samples = preProcessedSample
                    else:
                        df_samples = pd.concat([df_samples,preProcessedSample],ignore_index=True, axis=0)
                except Exception as e:
                    Utils.show(f"An error ocurred while preprocessing sample {row['label']}/{row['sampleID']}:{str(e)}",verbose)

            if df_samples is not None:
                Utils.show(f"Saving pre-processed features on {output}/{label}_pre_processed.csv",verbose)
                df_samples.to_csv(f"{output}/{label}_pre_processed.csv",index=False)
                Utils.show('Saved!',verbose)
            else:
                Utils.show(f"Failed to pre-process dataset.",verbose)

        Utils.show(f"Loading live pre-processed Dataframe...",verbose)
        df_live = pd.read_csv(f"{output}/live_pre_processed.csv")
        Utils.show(f"Loaded!",verbose)
        Utils.show(f"Loading spoof pre-processed Dataframe...",verbose)
        df_spoof = pd.read_csv(f"{output}/spoof_pre_processed.csv")
        Utils.show(f"Loaded!",verbose)

        Utils.show(f"Merging datasets...",verbose)
        df_merged = pd.concat([df_live,df_spoof])
        Utils.show(f"Merged!",verbose)
        
        Utils.show(f"Saving merged pre-processed features on {output}/pre_processed.csv",verbose)
        df_merged.to_csv(f"{output}/pre_processed.csv",index=False)
        Utils.show('Saved!',verbose)
        
        return

    @staticmethod
    def __getPreProcessedSample(df_preProcessed,label,sampleID,verbose):
        preProcessedSample = None
        if df_preProcessed is not None:
            # get pre processed data
            preProcessedSample = df_preProcessed.loc[(df_preProcessed["label"] == label) & (df_preProcessed["sampleID"] == sampleID)]
            if len(preProcessedSample) == 0:
                preProcessedSample = None
                Utils.show(f"Failed to restore pre-computed data for sample {label}/{sampleID}",verbose)
        return preProcessedSample
    

    @staticmethod
    def selectFrames(df,sourcePath,N,faceDetector=None,landmarkExtractor=None,df_preProcessed=None,verbose=False):
        
        selectedFrames = []
        for _,sample in df.iterrows():
            try:
                # pick frames
                samplePath = f"{sourcePath}/{sample['label']}/{sample['sampleID']}"
                df_sample = Model.__getPreProcessedSample(df_preProcessed,sample["label"],sample["sampleID"],verbose)
                selection = FrameSelector.selectFrames(samplePath,N,faceDetector,landmarkExtractor,df_sample,verbose)
                
                # store [label,sampleID,frame_ref,frame_0,...,frame_N-1]
                selectedFrames.append([sample['label'],sample['sampleID']] + selection)
                
            except Exception as e:
                Utils.show(f"[Frame selection] Error: {e}",verbose)

        return pd.DataFrame(selectedFrames,columns=['label','sampleID','frame_ref']+[f'frame_{i}' for i in range(0,N)])
    
    
    @staticmethod
    def extractFeatures(df_frames,sourcePath,N,faceDetector,landmarkExtractor,df_preProcessed=None,verbose=False):
        
        extractedFeatures = []
        distortionVectorSize = 0
        
        for _,sample in df_frames.iterrows():
            try:
                samplePath = f"{sourcePath}/{sample['label']}/{sample['sampleID']}"
                df_sample = Model.__getPreProcessedSample(df_preProcessed,sample["label"],sample["sampleID"],verbose)
                # extract distortion features
                distortionFeatures = FeatureExtractor.getDistortionFeatures(samplePath,sample,N,faceDetector,landmarkExtractor,df_sample,verbose)
                distortionVectorSize = len(distortionFeatures)
                # store [label,sampleID,distortion_feature_0,...,distortion_feature_K]
                extractedFeatures.append([sample['label'],sample['sampleID']] + distortionFeatures)

            except Exception as e:
                Utils.show(f"[Feature extraction] Error: {e}",verbose)

        return pd.DataFrame(extractedFeatures,columns=['label','sampleID']+[f"distortion_feature_{i}" for i in range(0,distortionVectorSize)])
    

    @staticmethod
    def train(df_train,df_val,learningRate=0.1,batchSize=50,numEpochs=1000,N=7,savePath="./faceCloseUp_weights",verbose=False):
        
        # Load trainig and validation data
        Utils.show("Loading train/val partitions...",verbose)
        train_dataset = CloseUpLivenessDataset(df_train,transform=ToTensor(),N=N)
        val_dataset = CloseUpLivenessDataset(df_val,transform=ToTensor(),N=N)
        Utils.show("Loading complete!!\n",verbose)

        # instantiate model and train
        Utils.show("Training Face CloseUp method!\n",verbose)
        
        Utils.show(f"Learning Rate: {learningRate}",verbose)
        Utils.show(f"Batch size: {batchSize}",verbose)
        Utils.show(f"Num epochs: {numEpochs}",verbose)
        
        # Instantiate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Classifier(N=N,distortionFeatureSize=2278)
        model.to(device)
        # Define the loss function and optimizer
        criterion = torchNN.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learningRate)
        # Creating loaders
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        # Training loop
        Utils.show("Training loop...",verbose)
        best_hter = float('inf')
        for epoch in range(numEpochs):
            Utils.show("\n",verbose)
            # Train one epoch
            model.train()
            loss,accuracy,hter,f1 = run_one_epoch(model,train_loader,criterion,optimizer,splitInputs=False,device=device)
            Utils.printEpoch("Train",epoch+1,loss,accuracy,hter,f1,verbose)

            # Check validation
            val_hter = float('inf')
            with torch.no_grad():
                model.eval()  # Set the model to evaluation mode
                val_loss,val_acc,val_hter,val_f1 = run_one_epoch(model,val_loader,torchNN.BCEWithLogitsLoss(),None,splitInputs=False,device=device)
                Utils.printEpoch("Val",epoch+1,val_loss,val_acc,val_hter,val_f1,verbose)

            # Save the model weights if the current HTER is better than the best HTER
            if val_hter < best_hter and savePath is not None:
                best_hter = val_hter
                torch.save(model.state_dict(), savePath)
                Utils.show(f'Saved weights at epoch {epoch + 1} with HTER: {val_hter:.4f}%',verbose)
                
        Utils.show("Finished!",verbose)
        return
    

    @staticmethod
    def eval(df_test,N=7,weightsPath="",verbose=False):
        # Load trainig and validation data
        Utils.show("Loading test partition...",verbose)
        test_dataset = CloseUpLivenessDataset(df_test,transform=ToTensor(),N=N)
        Utils.show("Loading complete!!\n",verbose)

        # instantiate model and train
        Utils.show("Testing Camera CloseUp method!\n",verbose)
        
        # Instantiate model and load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Classifier(N=N,distortionFeatureSize=2278)
        model.to(device)
        model.loadWeights(weightsPath)
        
        # Creating loader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            loss,accuracy,hter,f1 = run_one_epoch(model,test_loader,torchNN.BCEWithLogitsLoss(),None,splitInputs=False,device=device)
            Utils.printEpoch("Eval",0,loss,accuracy,hter,f1,True)

        Utils.show("Finished!",verbose)