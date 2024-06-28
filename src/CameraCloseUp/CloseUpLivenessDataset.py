from torch.utils.data import Dataset
import numpy as np
    

class CloseUpLivenessDataset(Dataset):
    def __init__(self, df, transform=None,N=24,distortionFeatureSize=2088):
        self.source_df = df
        self.transform = transform
        self.classes = df["label"].unique().tolist()
        # Fix live class at position 1
        if any([c == "live" for c in self.classes]):
            self.classes.remove("live")
            self.classes.insert(1,"live")
        # Load dataframe
        self.data = self._load_data(N,distortionFeatureSize)

    def _load_data(self,N,distortionFeatureSize):
        data = []

        allCols = self.source_df.columns.tolist()
        distortionFeatureSize = len([s for s in allCols if s.startswith("distortion_feature_")])
        distortionFeatureCols = [f"distortion_feature_{i}" for i in range(0,distortionFeatureSize)]

        for _,sample_row in self.source_df.iterrows():
            label = self.classes.index(sample_row["label"])

            rawDistortionFeatures = sample_row[distortionFeatureCols].tolist()
            distortionFeatures = np.array(rawDistortionFeatures, dtype=np.float32).reshape((N ,int(distortionFeatureSize // N)))
            data.append((distortionFeatures, label))
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        distortionFeatures, label = self.data[idx]

        if self.transform:
            distortionFeatures = self.transform(distortionFeatures)
        
        return distortionFeatures, label
    