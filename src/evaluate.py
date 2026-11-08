
import torch

def evaluate_model( model, test_loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.view(image.size(0),-1).to(device)
            label = label.to(device)

            output = model(image)
            predictions = output.argmax(dim=1)

            correct += (predictions == label).sum().item()
            total += label.size(0)

    accuracy = 100*(correct/total)
    return accuracy