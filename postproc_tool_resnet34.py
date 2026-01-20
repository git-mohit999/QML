import torch
import torchvision.models as models
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn as nn
from torchvision.transforms import GaussianBlur
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
# shape check
X, _ = next(iter(test_dl))
X = X.to(device)
out = model(X)
print(out.shape)

# shape check
X, _ = next(iter(test_dl))
X = X.to(device)
out = model(X)
print(out.shape)

# which layers are frozen (if you freeze later)
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name)
        break

"""""""""
weights = models.ResNet34_Weights.DEFAULT
model = models.resnet34(weights=weights)
model.fc = torch.nn.Identity()#was for classification of imagenet so removed it operationally
#print(model) - get model archi
model.to(device)


dataset_vis = CIFAR100(
    root=r"D:/1/qmxir/CGAP_diff/datasets",
    train=True,
    download=False,
    transform=weights.transforms()
)

class_names = dataset_vis.classes
image,label = dataset_vis[0]
#img_plot = image.permute(1, 2, 0)#reorder to  H,W,C
#plt.imshow(img_plot,cmap = "gray")
#plt.title(label = class_names[label])
#plt.show()
img = image.unsqueeze(0).to(device)

#other images for processing
""""""
# original PIL image from CIFAR (before transforms)
pil_img, label = CIFAR100(
    root=r"D:/1/qmxir/CGAP_diff/datasets",
    train=True,
    download=False
)[0]

# negative in pixel space
pil_neg = F.invert(pil_img)

# now apply ImageNet transforms
img_neg = weights.transforms()(pil_neg).unsqueeze(0).to(device)
""""""

blur = GaussianBlur(kernel_size=11, sigma=5.0)

# blur BEFORE normalization
pil_blur = blur(pil_img)

# apply ImageNet transforms
img_blur = weights.transforms()(pil_blur).unsqueeze(0).to(device)
""""""
perm = torch.tensor([0,1,2])
img_perm = img[:, perm, :, :]
""""""

model = model.eval()
backbone = nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4

).to(device)

tail = nn.Sequential(
    

    
    model.avgpool,
    nn.Flatten()
).to(device)

backbone.eval()
tail.eval()

# Original forward pass
with torch.no_grad():
    feats = backbone(img)      # [1, C, H, W]
    z_orig = tail(feats)       # [1, D]

seed = 8769786
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

C = feats.shape[1]
D = 16

E = torch.randn(C, D, device=feats.device)
E = torch.nn.functional.normalize(E, dim=0)

def rep_cap_pr(E, x_i):
    psi = E.T @ x_i
    psi = psi / torch.norm(psi)
    p = psi**2
    return 1.0 / (p**2).sum()



def rep_cap(E,x_i):
    psi = E.T@x_i
    psi = psi/torch.norm(psi)
    p = psi ** 2
    return -(p * torch.log2(p + 1e-8)).sum()

#x_i is an image region
# Mean feature vector for occlusion
mean_vec = feats.mean(dim=(2, 3), keepdim=True)

_, C, H, W = feats.shape
importance_map = torch.zeros(H, W, device=feats.device)
cap_map = torch.zeros(H, W, device=feats.device)

for i in range(H):
    for j in range(W):
        feats_occ = feats.clone()
        feats_occ[:, :, i, j] = mean_vec.squeeze(-1).squeeze(-1)
        #feats_occ[:, :, i, j] = 0

        with torch.no_grad():
            z_occ = tail(feats_occ)

        # Importance = change in final representation
        importance_map[i, j] = torch.norm(z_orig - z_occ)

        x_i = feats[0, :, i, j] 
        #cap_map[i, j] = rep_cap(E, x_i)
        cap_map[i, j] = rep_cap_pr(E, x_i)

# Visualization
plt.figure(figsize=(10,4))

importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
cap_map = (cap_map - cap_map.min()) / (cap_map.max() - cap_map.min() + 1e-8)
"""
plt.subplot(1,2,1)
plt.imshow(importance_map.detach().cpu(), cmap="hot")
plt.colorbar()
plt.title("Occlusion Importance")

plt.subplot(1,2,2)
plt.imshow(cap_map.detach().cpu(), cmap="hot")
plt.colorbar()
plt.title("Representational Capacity")

plt.show()

"""
diff_map = importance_map - cap_map

plt.imshow(diff_map.detach().cpu(), cmap="bwr")
plt.colorbar()
plt.title("Importance - Capacity")
plt.show()

