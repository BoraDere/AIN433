# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

import os
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# %%
DATA_PATH = os.path.join(os.getcwd(), 'food11')
TEST_PATH = os.path.join(DATA_PATH, 'test')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VAL_PATH = os.path.join(DATA_PATH, 'validation')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class FoodDataset(Dataset):
    def __init__(self, path, transformations=None, train=True, normalize=None):
        self.path = path
        self.transformations = transformations
        self.train = train
        self.samples = []
        self.normalize = normalize

        for i, label in enumerate(os.listdir(path)):
            label_path = os.path.join(path, label)
            for img in os.listdir(label_path):
                self.samples.append((os.path.join(label_path, img), i))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transformations:
            img = self.transformations(img)
        return img, label

def calculate_dataset_stats(dataset_loader):
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_pixels = 0

    print("Calculating dataset statistics...")
    for images, _ in tqdm(dataset_loader):
        batch_size, _, height, width = images.shape
        num_pixels += batch_size * height * width
        
        channels_sum += torch.sum(images, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(images ** 2, dim=[0, 2, 3])
    
    mean = channels_sum / num_pixels
    std = torch.sqrt(channels_squared_sum / num_pixels - mean ** 2)
    
    return mean.tolist(), std.tolist()

def create_dataloaders(batch_size=32, random_seed=42):
    torch.manual_seed(random_seed)

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    temp_train = FoodDataset(TRAIN_PATH, transformations=base_transform, train=True)
    temp_train_loader = DataLoader(temp_train, batch_size=batch_size, shuffle=True)

    mean, std = calculate_dataset_stats(temp_train_loader)
    print(f"Mean: {mean}, Std: {std}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = FoodDataset(TRAIN_PATH, transformations=train_transform, train=True, normalize=(mean, std))
    test_dataset = FoodDataset(TEST_PATH, transformations=test_transform, train=False, normalize=(mean, std))
    val_dataset = FoodDataset(VAL_PATH, transformations=test_transform, train=False, normalize=(mean, std))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader

train_loader_32, test_loader_32, val_loader_32 = create_dataloaders()
train_loader_64, test_loader_64, val_loader_64 = create_dataloaders(batch_size=64)

# %%
class StandardCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(StandardCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class StandardCNNwDropout(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.5):
        super(StandardCNNwDropout, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # residual connection
        out = F.relu(out)
        
        return out

class ResidualCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(ResidualCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res_block1 = self._create_res_block(64, 128, stride=1)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.res_block2 = self._create_res_block(256, 256, stride=1)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def _create_res_block(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResidualBlock(in_channels, out_channels, stride, downsample)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.res_block1(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.res_block2(x)
        
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class ResidualCNNwDropout(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.5):
        super(ResidualCNNwDropout, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res_block1 = self._create_res_block(64, 128, stride=1)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.res_block2 = self._create_res_block(256, 256, stride=1)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def _create_res_block(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResidualBlock(in_channels, out_channels, stride, downsample)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.res_block1(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.res_block2(x)
        
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

# %%
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training..."):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating..."):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_predictions, all_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=DEVICE):
    best_acc = 0.0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Training complete! Best validation accuracy:", best_acc)
    return model

def test_model(model, test_loader, criterion, device):
    test_loss, test_acc, predicted, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    predicted = predicted.cpu()
    labels = labels.cpu()
    
    cm = confusion_matrix(labels, predicted)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return predicted, labels, cm

# %%
lr = 0.001
batch_size = 32

model = StandardCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

std_0001_32 = train_model(model, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0005
batch_size = 32

model = StandardCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

std_00005_32 = train_model(model, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0001
batch_size = 32

model = StandardCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

std_00001_32 = train_model(model, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.001
batch_size = 64

model = StandardCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

std_0001_64 = train_model(model, train_loader_64, val_loader_64, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0005
batch_size = 64

model = StandardCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

std_00005_64 = train_model(model, train_loader_64, val_loader_64, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0001
batch_size = 64

model = StandardCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

std_00001_64 = train_model(model, train_loader_64, val_loader_64, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
best_std_cnn = std_00001_32

# %%
lr = 0.001
batch_size = 32

model = ResidualCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

res_0001_32 = train_model(model, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0005
batch_size = 32

model = ResidualCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

res_00005_32 = train_model(model, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0001
batch_size = 32

model = ResidualCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

res_00001_32 = train_model(model, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.001
batch_size = 64

model = ResidualCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

res_0001_64 = train_model(model, train_loader_64, val_loader_64, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0005
batch_size = 64

model = ResidualCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

res_00005_64 = train_model(model, train_loader_64, val_loader_64, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0001
batch_size = 64

model = ResidualCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

res_00001_64 = train_model(model, train_loader_64, val_loader_64, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
best_res_cnn = res_00001_64

# %%
predicted_std, labels_std, cm_std = test_model(best_std_cnn, test_loader_32, criterion, DEVICE)

# %%
predicted_res, labels_res, cm_std = test_model(best_res_cnn, test_loader_64, criterion, DEVICE)

# %% [markdown]
# ## Dropout

# %%
lr = 0.0001
batch_size = 32

model = StandardCNNwDropout().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

std_dropout_00001_32 = train_model(model, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
lr = 0.0001
batch_size = 64

model = ResidualCNNwDropout().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

res_dropout_00001_64 = train_model(model, train_loader_64, val_loader_64, criterion, optimizer, num_epochs=50, device=DEVICE)

# %%
class ShuffleNetTransfer(nn.Module):
    def __init__(self, num_classes=11, freeze_layers=True, unfreeze_conv_layers=0):
        super(ShuffleNetTransfer, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False
                
            for param in self.model.fc.parameters():
                param.requires_grad = True
            
            if unfreeze_conv_layers > 0:
                layers_to_unfreeze = ['conv5']
                if unfreeze_conv_layers > 1:
                    layers_to_unfreeze.append('stage4')
                
                for name, param in self.model.named_parameters():
                    for layer in layers_to_unfreeze:
                        if layer in name:
                            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)

def count_parameters(model):
    params_to_update = []
    names_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            names_to_update.append(name)
    
    print(f"Training {len(params_to_update)}/{len(list(model.parameters()))} parameters")
    print(f"Trainable layers: {names_to_update}")
    return params_to_update

# %%
model_fc_only = ShuffleNetTransfer(num_classes=11, freeze_layers=True, unfreeze_conv_layers=0).to(DEVICE)

trainable_params = count_parameters(model_fc_only)

lr = 0.0001
optimizer = optim.AdamW(trainable_params, lr=lr)
criterion = nn.CrossEntropyLoss()

fc_only_model = train_model(model_fc_only, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

fc_only_predicted, fc_only_labels, fc_only_cm = test_model(fc_only_model, test_loader_32, criterion, DEVICE)

# %%
model_fc_conv = ShuffleNetTransfer(num_classes=11, freeze_layers=True, unfreeze_conv_layers=2).to(DEVICE)

trainable_params = count_parameters(model_fc_conv)

lr = 0.0001
optimizer = optim.AdamW(trainable_params, lr=lr)
criterion = nn.CrossEntropyLoss()

fc_conv_model = train_model(model_fc_conv, train_loader_32, val_loader_32, criterion, optimizer, num_epochs=50, device=DEVICE)

fc_conv_predicted, fc_conv_labels, fc_conv_cm = test_model(fc_conv_model, test_loader_32, criterion, DEVICE)

# %% [markdown]
# ### Test dropout models

# %%
predicted_std_dropout, labels_std_dropout, cm_std_dropout = test_model(std_dropout_00001_32, test_loader_32, criterion, DEVICE)

# %%
predicted_res_dropout, labels_res_dropout, cm_std_dropout = test_model(res_dropout_00001_64, test_loader_64, criterion, DEVICE)


