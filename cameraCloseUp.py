import argparse
import pandas as pd
from src.CameraCloseUp.Model import Model
from src.Utils.Utils import Utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-process, Train and evaluate the Camera CloseUp method')
    
    parser.add_argument('--pre_processed', required=False,type=str, default=None,help="Path to pre-processed csv file")
    parser.add_argument('--dataset', required=False,type=str, default=None,help="Path to dataset")
    parser.add_argument('--dlib_dat_files', required=False,type=str, default=None,help="Path to directory containing shape_predictor_68_face_landmarks.dat and mmod_human_face_detector.dat")
    parser.add_argument('--weights', required=False,type=str, default=None,help="Path to pre trained weights")

    parser.add_argument('--train', required=False,type=str, default=None,help="Path to train csv")
    parser.add_argument('--test', required=False,type=str, default=None,help="Path to test csv")
    parser.add_argument('--val', required=False,type=str, default=None,help="Path to validation csv")

    parser.add_argument('--N', required=False,type=int, default=24,help="Amount of sampled frames")
    parser.add_argument('--b', required=False,type=int, default=3,help="Amount of bins")

    parser.add_argument('--lr', required=False,type=float, default=0.001,help="Learning rate")
    parser.add_argument('--bs', required=False,type=int, default=50,help="Batch size")
    parser.add_argument('--epoch', required=False,type=int, default=500,help="Num epochs")

    parser.add_argument('--v',action="store_true",help="Verbose")

    parser.add_argument('--process_dataset',action="store_true",help="Pre-process the entire dataset")
    parser.add_argument('--select_frames',action="store_true",help="To compute frame selection")
    parser.add_argument('--extract_features',action="store_true",help="To compute landmarks distance vectors")
    parser.add_argument('--learn',action="store_true",help="To train the Model model")
    parser.add_argument('--eval',action="store_true",help="To evaluate the model on test set")
    args = parser.parse_args()

    # Load variables models/dataframes
    faceDetector = None
    landmarkExtractor = None
    if (args.select_frames or args.extract_features or args.process_dataset) and args.dlib_dat_files is not None:
        faceDetector = Utils.loadFaceDetector(args.dlib_dat_files)
        if faceDetector is None:
            print("Failed to load face detector!")
            exit()
        landmarkExtractor = Utils.loadLandmarkExtractor(args.dlib_dat_files)
        if landmarkExtractor is None:
            print("Failed to load face detector!")
            exit()


    if args.process_dataset:
        if args.dataset is None or args.pre_processed is None or args.dlib_dat_files is None:
            print("Error: missing too few arguments!\n The arguments --dataset,--dlib_dat_files and --pre_processed must be defined.")
            exit()
        
        # read dataset annotation file
        df_annotations = pd.read_csv(f"{args.dataset}/annotations/annotations.csv")
        # pre-process the entire dataset
        Model.preProcessDataset(df_annotations,args.dataset,faceDetector,landmarkExtractor,args.pre_processed,args.v)

        print("Finished pre-processing step!")
        print("Exiting...")
        exit()


    df_preProcessed = None
    if args.pre_processed is not None and (args.select_frames or args.extract_features):
        df_preProcessed = pd.read_csv(args.pre_processed)


    if args.select_frames:
        if args.dataset is None or args.train is None or args.test is None or args.val is None or args.dlib_dat_files is None:
            print("Error: missing too few arguments!\n The arguments --dataset,--dlib_dat_files,--train,--test and --val must be defined.")
            exit()

        # read dataset annotation file
        df_annotations = pd.read_csv(f"{args.dataset}/annotations/annotations.csv")

        # read list of samples for each partition
        df_train_sampleID = pd.read_csv(args.train)
        df_test_sampleID = pd.read_csv(args.test)
        df_val_sampleID = pd.read_csv(args.val)

        # select the annotated data for each partition
        df_train = pd.merge(df_annotations,df_train_sampleID,how="inner",on=["label","sampleID"])
        df_test = pd.merge(df_annotations,df_test_sampleID,how="inner",on=["label","sampleID"])
        df_val = pd.merge(df_annotations,df_val_sampleID,how="inner",on=["label","sampleID"])
        
        # select frames
        for partition,df in zip(["train","test","val"],[df_train,df_test,df_val]):
            
            print(f"Selecting frames from {partition}")
            df_frames = Model.selectFrames(df,args.dataset,args.N,args.b,faceDetector,landmarkExtractor,df_preProcessed,args.v)
            if df_frames is None:
                exit()
            else:
                print(f"Saving selected frames in frames_{partition}...")
                df_frames.to_csv(f"frames_{partition}.csv",index=False)
                print(f"Done!")

        if args.extract_features is None:
            exit()
        else:
            args.train = "frames_train.csv"
            args.test = "frames_test.csv"
            args.val = "frames_val.csv"


    if args.extract_features:
        if args.dataset is None or args.train is None or args.test is None or args.val is None or args.dlib_dat_files is None:
            print("Error: missing too few arguments!\n The arguments --dataset,--dlib_dat_files,--train,--test,--val must be defined.")
            exit()

        df_train = pd.read_csv(args.train)
        df_test = pd.read_csv(args.test)
        df_val = pd.read_csv(args.val)

        for partition,df in zip(["train","test","val"],[df_train,df_test,df_val]):
            
            print(f"Extracting features from {partition}...")
            df_features = Model.extractFeatures(df,args.dataset,args.N,faceDetector,landmarkExtractor,df_preProcessed,args.v)
            if df_features is None:
                exit()
            else:
                print(f"Saving features in features_{partition}...")
                df_features.to_csv(f"features_{partition}.csv",index=False)
                print(f"Done!")
            
        if args.learn is None:
            exit()
        else:
            args.train = "features_train.csv"
            args.test = "features_test.csv"
            args.val = "features_val.csv"

    if args.learn:
        if args.train is None is None or args.val is None:
            print("Error: missing too few arguments!\n The arguments --dataset,--train,--val must be defined.")
            exit()

        df_train = pd.read_csv(args.train)
        df_val = pd.read_csv(args.val)

        Model.train(df_train,df_val,args.lr,args.bs,args.epoch,args.N,"./cameraCloseUp_weights.pth",args.v)
        
    if args.eval:
        if args.weights is None or args.test is None:
            print("Error: missing too few arguments!\n The arguments --weights and --test must be defined.")
            exit()

        df_test = pd.read_csv(args.test)
        Model.eval(df_test,args.N,args.weights,verbose=args.v)
        
