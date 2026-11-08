from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__) .parent.parent
datafile = BASE_DIR /"data"

def load_data():
    transform = transforms.ToTensor()

    train_data = MNIST(
        root=datafile,
        train= True,
        download= True,
        transform= transform
    )

    test_data = MNIST(
        root=datafile,
        train= False,
        download= True,
        transform= transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size= 64,
        shuffle= True
    )

    test_loader = DataLoader(
        test_data,
        batch_size= 64,
        shuffle= False
    )

    return train_loader, test_loader, test_data

if __name__ == "__main__":
    train_loader, test_loader, test_data = load_data()

    print(">> Dataset loaded successfully <<")
    print(f"Training Images:{len(train_loader.dataset)}")
    print(f"Testing Images:{len(test_loader.dataset)}")