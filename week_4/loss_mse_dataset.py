# Convert targets to one-hot encoding
def one_hot_encode(targets, num_classes=10):
    return torch.eye(num_classes)[targets]

# Modify the training loop
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        target_one_hot = one_hot_encode(target)  # Convert target to one-hot
        loss = criterion(torch.exp(output), target_one_hot)  # Use exp(output) to invert LogSoftmax
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    # Continue as before