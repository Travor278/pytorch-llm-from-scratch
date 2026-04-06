import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch version: {torch.__version__}")
print(f"selected device: {device}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda build: {torch.version.cuda}")
print(f"gpu count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu name: {torch.cuda.get_device_name(0)}")
