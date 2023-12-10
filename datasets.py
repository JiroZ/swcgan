from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1


# The SWCGANDataset dataset module.
class SWCGANDataset(Dataset):
    def __init__(self, root_dir, transform_gen, transform_real):
        self.root_dir = root_dir
        self.transform_gen = transform_gen
        self.transform_real = transform_real
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert("RGB")

        img_low_res = self.transform_gen(image)
        img_real = self.transform_real(image)

        return img_low_res, img_real

    # Define transformations for the generator (64x64) and real images (256x256)


# Prepare the datasets.
def get_datasets(
        train_image_paths, train_label_paths,
        valid_image_path, valid_label_paths
):
    transform_gen = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # Add other transformations if needed
    ])

    transform_real = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Add other transformations if needed
    ])

    dataset_train = SWCGANDataset(
        train_image_paths, train_label_paths
    )
    dataset_valid = SWCGANDataset(
        valid_image_path, valid_label_paths
    )
    return dataset_train, dataset_valid


# Prepare the data loaders
def get_dataloaders(dataset_train, dataset_valid):
    train_loader = DataLoader(
        dataset_train,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )
    return train_loader, valid_loader
