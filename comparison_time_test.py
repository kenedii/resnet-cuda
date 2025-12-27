import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet34
from torch.utils.data import DataLoader
import time
import psutil
import os

# Import custom CUDA extension
import cu_resnet

# -------------------------------------------------------------
# Global optimization: let cuDNN pick the fastest algorithms
# -------------------------------------------------------------
torch.backends.cudnn.benchmark = True

# Data loaders for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

# ================================================================
# Custom fused Conv + BN + ReLU module 
# ================================================================
class CustomFusedConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 3, 3, device=device)
        )
        self.bias = nn.Parameter(
            torch.randn(out_channels, device=device)
        )
        self.gamma = nn.Parameter(
            torch.ones(out_channels, device=device)
        )
        self.beta = nn.Parameter(
            torch.zeros(out_channels, device=device)
        )
        # Single float32 running stats
        self.register_buffer(
            'running_mean',
            torch.zeros(out_channels, device=device)
        )
        self.register_buffer(
            'running_var',
            torch.ones(out_channels, device=device)
        )

    def forward(self, x):
        if self.training:
            save_mean = torch.zeros(self.out_channels, dtype=torch.float32, device=x.device, requires_grad=False)
            save_invvar = torch.zeros(self.out_channels, dtype=torch.float32, device=x.device, requires_grad=False)
        else:
            save_mean = torch.empty(0, device=x.device)
            save_invvar = torch.empty(0, device=x.device)

        out = cu_resnet.fused_conv_bn_relu_forward(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.running_mean, self.running_var,
            save_mean, save_invvar,
            self.stride, self.padding,
            1e-5, 0.1, self.training
        )
        return out

# ================================================================
# BasicBlock using custom fused layer
# ================================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = CustomFusedConvBnRelu(
            in_planes, planes, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out, inplace=True)
        return out

# ================================================================
# ResNet34
# ================================================================
class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)
    def _make_layer(self, planes, blocks, stride):
        layers = []
        for i in range(blocks):
            layers.append(
                BasicBlock(self.in_planes, planes, stride if i == 0 else 1)
            )
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# ================================================================
# Benchmark
# ================================================================
def run_experiment(name, model):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    start_mem = get_memory_mb()
    start_time = time.time()
    print(f"\n=== {name} ===")
    for epoch in range(15):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        acc = 100. * correct / total
        print(
            f"Epoch {epoch+1:2d} | "
            f"Loss: {epoch_loss/len(train_loader):.4f} | "
            f"Acc: {acc:.2f}%"
        )
    train_time = time.time() - start_time
    peak_mem = get_memory_mb() - start_mem

    # Inference timing (use first full batch after warm-up)
    model.eval()
    inference_times = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            # Warm-up run
            _ = model(data)
            torch.cuda.synchronize()
            start = time.time()
            _ = model(data)
            torch.cuda.synchronize()
            end = time.time()
            inference_times.append(end - start)
            break  # Only need one full batch

    avg_inf_time_per_100 = (sum(inference_times) / len(inference_times)) * (100 / 128) * 1000

    print(f"\n{name} Results:")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Peak memory increase: {peak_mem:.1f} MB")
    print(f"  Avg inference time (100 examples): {avg_inf_time_per_100:.2f} ms")
    print("-" * 60)

# ================================================================
# Run all three experiments
# ================================================================
try:
    # 1. Your custom fused model
    run_experiment(
        "1. Custom Fused Forward + Torch Autograd",
        ResNet34()
    )

    # 2. Standard stock PyTorch ResNet34
    stock_model = resnet34(num_classes=10)
    stock_model.conv1 = nn.Conv2d(
        1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    stock_model.maxpool = nn.Identity()
    run_experiment(
        "2. Stock PyTorch ResNet34",
        stock_model
    )

    # 3. Stock PyTorch ResNet34 + torch.compile() (state-of-the-art)
    compiled_model = resnet34(num_classes=10)
    compiled_model.conv1 = nn.Conv2d(
        1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    compiled_model.maxpool = nn.Identity()
    compiled_model = compiled_model.to(device)

    print("Compiling model with torch.compile(mode='reduce-overhead')... (may take 30-60s on first run)")
    compiled_model = torch.compile(compiled_model, mode="reduce-overhead")

    run_experiment(
        "3. Stock PyTorch ResNet34 + torch.compile()",
        compiled_model
    )

finally:
    cu_resnet.destroy_libs()
