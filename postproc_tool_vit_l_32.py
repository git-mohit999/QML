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
weights = models.ViT_L_32_Weights.DEFAULT
model = models.vit_l_32(weights=weights)
model.head = torch.nn.Identity()#was for classification of imagenet so removed it operationally
#print(model) #- get model archi
model.to(device)


dataset_vis = CIFAR100(
    root=r"D:/1/qmxir/CGAP_diff/datasets",
    train=True,
    download=False,
    transform=weights.transforms()
)

class_names = dataset_vis.classes
image,label = dataset_vis[0]
img_plot = image.permute(1, 2, 0)#reorder to  H,W,C
#plt.imshow(img_plot,cmap = "gray")
#plt.title(label = class_names[label])
plt.show()
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
to_occlulde = False
active_block = None
Batch = 4
#both of these are callbacks
def pre_hook(module, inputs):
    if not to_occlulde:
        return None
    
    x = inputs[0].clone()          # (B, N, D)

    # split
    cls = x[:, :1, :]              # (B, 1, D)
    patches = x[:, 1:, :]          # (B, P, D)
    B, P, D = patches.shape
    # collapse patches to mean
    
    start = active_block * Batch
    end = min(start + Batch, P)#for last block to avoid indexing error
    block = patches[:, start:end, :]                  # (B, K, D)
    block_mean = block.mean(dim=1, keepdim=True)      # (B, 1, D)

    patches[:, start:end, :] = block_mean.expand(-1, end - start, -1)
    # reassemble
    x = torch.cat([cls, patches], dim=1)

    return (x,)   # must return a tuple

feats = {}
def post_hook(module, input, output):
    feats["layer11"] = output

h1 = model.encoder.layers[7].register_forward_pre_hook(pre_hook)
h2 = model.encoder.layers[7].register_forward_hook(post_hook)


# Original forward pass
with torch.no_grad():
    feat = model(img)

feat = feats['layer11'] #(B,N_Tokens,1024)

cls_feat = feat[:,0]#global semantic array
patch_feat = feat[:,1:]#region level semnatic array
z_orig = cls_feat
num_blocks = patch_feat.shape[1]// Batch + (1 if patch_feat.shape[1]% Batch !=0 else 0)

seed = 8769786
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

C = patch_feat.shape[2]
D = C

E = torch.randn(C, C, dtype=torch.float, device=patch_feat.device)
E = torch.linalg.qr(E).Q #E=QR, Q is one with orthonormal columns which is the unitary i want for embedding

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
#Representation capacity tells you how much of the available feature space the ViT effectively uses at that layer to encode the image in the CLS token.
depth_cls = rep_cap_pr(E, z_orig.T).item()


# Mean feature vector for occlusion, ViT : using cls and patch tokens, checking if chsnging patch tokens affect the cls token:
# dtermine importance of each patch


#since tokens, gonna make a bar graph of importance and representaio capacity given to per patch token
B,N_t,num = patch_feat.shape
importance_map = torch.zeros(num_blocks, device=patch_feat.device)

to_occlulde = True

for i in range(num_blocks):

        active_block = i
        with torch.no_grad():
            feat_occ = model(img)
        # Importance = change in final representation
        feat_occ = feats['layer11']
        cls_feat_occ = feat_occ[:, 0,:]  # global semantic array
        patch_feat_occ = feat_occ[:, 1:,:]  # region level semnatic array
        z_occ = cls_feat_occ
        importance_map[i] = torch.norm(z_orig - z_occ,dim=1)

active_block = None
h1.remove()
h2.remove()

# Visualization
importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)


plt.bar(torch.arange(num_blocks),importance_map)
plt.xlabel("patch block number(block size is 4 patch tokens)")
plt.ylabel("relative importance")
plt.title("Occlusion Importance per blck size of size 4 patch tokens wrt CLS token")
plt.show()
print(f"Representation capacity at CLS token: {depth_cls:.4f}")

