#patient_id,image_path,mask_path,mask 

import csv
from torch.utils.data import DataLoader



with open("Healthcare_AI_Datasets/Brain_MRI/data_mask.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        image_path = 
        print(line[3])


#The image patch has to be Brain_Tumor_Detector/Healthcare_AI_Datasets/Brain_MRI/img_path
class Dataset():

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# # split the data into train and test data

# from sklearn.model_selection import train_test_split

# train, test = train_test_split(brain_df_train, test_size = 0.15)

# # create a image generator
# from keras_preprocessing.image import ImageDataGenerator

# # Create a data generator which scales the data from 0 to 1 and makes validation split of 0.15
# datagen = ImageDataGenerator(rescale=1./255., validation_split = 0.15)

# train_generator=datagen.flow_from_dataframe(
# dataframe=train,
# directory= './',
# x_col='image_path',
# y_col='mask',
# subset="training",
# batch_size=16,
# shuffle=True,
# class_mode="categorical",
# target_size=(256,256))


# valid_generator=datagen.flow_from_dataframe(
# dataframe=train,
# directory= './',
# x_col='image_path',
# y_col='mask',
# subset="validation",
# batch_size=16,
# shuffle=True,
# class_mode="categorical",
# target_size=(256,256))

# # Create a data generator for test images
# test_datagen=ImageDataGenerator(rescale=1./255.)

# test_generator=test_datagen.flow_from_dataframe(
# dataframe=test,
# directory= './',
# x_col='image_path',
# y_col='mask',
# batch_size=16,
# shuffle=False,
# class_mode='categorical',
# target_size=(256,256))

