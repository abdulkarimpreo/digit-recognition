
def train_model(model, optimizer, loss_fn, train_loader, device):

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.view(images.size(0),-1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss +=loss.item()

            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")