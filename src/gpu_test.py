import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[TRAIN] device : ", device)
