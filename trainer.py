import torch
import math
import torch.nn.functional as F
from torch.optim import AdamW
from vilip import ViLIP
from data_loader import get_dataloader

def train(model, train_loader, optimizer, device, epochs=5):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, captions in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(images, captions)

            labels = torch.full_like(logits, -1.0)
            labels.fill_diagonal_(1.0)

            loss_matrix = -F.logsigmoid(labels * logits)
            loss = loss_matrix.sum()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    device = torch.device("cpu")

    model = ViLIP(
        fasttext_model_path="pretrained/cc.vi.300.bin",
        embed_dim=256,
        init_temp= math.log(10.0),
        init_bias=-10.0,
    )

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Load training data
    train_loader = get_dataloader(
        captions_file_path="dataset/train/captions_preprocessed.txt",
        batch_size=8,
    )

    train(model, train_loader, optimizer, device, epochs=5)

    # Save trained model
    torch.save(model.state_dict(), "models/vilip.pt")
    print("Saved model to models/vilip.pt")