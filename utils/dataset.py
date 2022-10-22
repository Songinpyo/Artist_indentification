import cv2
from torch.utils.data import Dataset
from sklearn import preprocessing


class CustomDataset(Dataset):
    def __init__(self, df, augmentation):
        self.df = df
        self.df.artist = preprocessing.LabelEncoder().fit_transform(df.artist.values)
        self.augmentation = augmentation
        self.df.path = df.img_path.values
        self.df.labels = df.artist.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.labels[index]

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        return image, label