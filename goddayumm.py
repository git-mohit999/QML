import torch
import torchvision.models as models
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
#from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
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

"""
weights = models.ResNet34_Weights.DEFAULT
model = models.resnet34(weights=weights)
model.fc = torch.nn.Identity()#was for classification of imagenet so removed it operationally
#print(model) - get model archi
model.to(device)

transform = weights.transforms()

train_set = CIFAR100(root=r"D:/1/qmxir/CGAP_diff/datasets",
                     train=True,
                     transform=transform)

test_set = CIFAR100(root=r"D:/1/qmxir/CGAP_diff/datasets",
                     train=False,
                     transform=transform)


##############################################
"""check data
class_names = train_set.classes
class_to_idx = train_set.class_to_idx
image,label = train_set[0]
print(f"Shape of image {image.shape} : [C,H,W]")
print(f"image label : {class_names[label]}")
"""
###############################################
"""see images
dataset_vis = CIFAR100(
    root=r"D:/1/qmxir/CGAP_diff/datasets",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

image,label = dataset_vis[0]
img_plot = image.permute(1, 2, 0)#reorder to  H,W,C
plt.imshow(img_plot,cmap = "gray")
plt.title(label = class_names[label])
plt.show()
"""
###############################################


BATCH_SIZE = 32
train_dl = DataLoader(dataset = train_set , batch_size = BATCH_SIZE , shuffle = True)
test_dl = DataLoader(dataset = test_set, batch_size = BATCH_SIZE , shuffle = False)

#images, labels = next(iter(train_dl))
#print(images.shape, labels.shape)


"""
making pooling a learnable, confidence-gated operation that delays spatial compression so gradients can reinforce strong, 
informative kernel activations(through gradients) before abstraction is finalized.
"""
#delaying irreversible compression so the layer effectively gets more learning opportunity before being training is over
class CGAP(torch.nn.Module):
    def __init__(self,tau= 0.7):
        super().__init__()
        self.tau = tau
        self.last_gate = None
    
    def forward(self, x):
        # x: (B, C, H, W)
        stat = x.var(dim=(1, 2, 3))          # (B,)
        if stat.numel() > 1:
            stat = (stat - stat.mean()) / (stat.std() + 1e-6)  # normalize

        #CNNs learn only through gradients.
        #So anything you want the model to 'learn' must be differentiable and parameterized.
        #So using smooth sigmoid function and not hard if-else
        gate = torch.sigmoid(stat - self.tau)  # (B,)
        self.last_gate = gate.detach()
        gate = gate.unsqueeze(-1)  # (B, 1), pytorch broadcasts this to (B, C) so matmul is A-ok

        pooled = x.mean(dim=(-2, -1))        # (B, C)

        #to preserve, ill preserve the activation energy of the kernl of the region thats being preserved
        #(after this layer only scalar which decides class matters so just preserving channel strength)

        #in mid layer application, i need to setup an isomorphic mapping to preserve the local structure.
        """
        Final-layer preservation requires magnitude consistency, whereas mid layer preservation requires
        relation-preserving mappings that maintain local feature geometry, similar to metric inheritance in student teacher distillation.
        """
        #preserved = torch.sqrt((x ** 2).mean(dim=(-2, -1)) + 1e-6)#RMS value of each channel
        preserved = x.amax(dim=(-2, -1))#max value of each channel

        return gate * preserved + (1.0 - gate) * pooled
    """BackProp
    Through convolutional connectivity,
    preserving a hot channel causes stronger gradients to flow back through all 
    kernels that contributed to the corresponding spatial evidence.
    """


####################################################################################################################################
#add diffusion


#add CGAP+diffusion

####################################################################################################################################

#model.eval()

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
model.avgpool = CGAP()
tail = nn.Sequential(
    model.avgpool,
    nn.Flatten()
).to(device)


backbone.eval()
tail.eval()

image,label = test_set[0]
img_plot = image.permute(1, 2, 0)
plt.imshow(img_plot,cmap = "gray")
plt.show()

with torch.no_grad():
    feats = backbone(image.unsqueeze(0).to(device))
    z_orig = tail(feats)   

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


mean_vec = feats.mean(dim=(2, 3), keepdim=True)

_, C, H, W = feats.shape
importance_map = torch.zeros(H, W, device=feats.device)
cap_map = torch.zeros(H, W, device=feats.device)

for i in range(H):
    for j in range(W):
        feats_occ = feats.clone()
        
        feats_occ[:, :, i, j] = 0

        with torch.no_grad():
            z_occ = tail(feats_occ)


        importance_map[i, j] = torch.norm(z_orig - z_occ)

        x_i = feats[0, :, i, j] 
        
        cap_map[i, j] = rep_cap_pr(E, x_i)

# Visualization
plt.figure(figsize=(10,4))

importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
cap_map = (cap_map - cap_map.min()) / (cap_map.max() - cap_map.min() + 1e-8)


plt.subplot(1,2,1)
plt.imshow(importance_map.detach().cpu(), cmap="hot")
plt.colorbar()
plt.title("Occlusion Importance")

plt.subplot(1,2,2)
plt.imshow(cap_map.detach().cpu(), cmap="hot")
plt.colorbar()
plt.title("Representational Capacity")

plt.show()

diff_map = importance_map - cap_map

plt.imshow(diff_map.detach().cpu(), cmap="bwr")
plt.colorbar()
plt.title("Importance - Capacity")
plt.show()

"""
#embeddings = []
#labels = []
#gate_val = []
with torch.inference_mode():
    for X, y in test_dl:
        X = X.to(device)
        feats = backbone(X)
        z_orig = tail(feats)
        #embeddings.append(feats.cpu())
        #labels.append(y)

        #gates = model.avgpool.last_gate.cpu()
        #gate_val.extend(gates.tolist())
#for retrievel/kNN/diffusion i need [M,512] and i have N*[32,512], cat merges to [M,512]
embeddings = torch.cat(embeddings)  # [M, 512]
labels = torch.cat(labels)          # [M] = N*[32]

#using images in test set as queries and gallery both and obtaning similarity matrix,topk
embeddings = F.normalize(embeddings, dim=1)
S = torch.matmul(embeddings, embeddings.T)  #similiairty matrix(cosine distance bc of normalization) [M, M]
S.fill_diagonal_(-1e9)#remove self-matches not to 0 though      # [1, D]
"""
"""
def recallK(S, labels, k):
    query_labels = labels.unsqueeze(1)
    topk_ = S.topk(k, dim=1).indices      # [M, K]
    retrieved_labels = labels[topk_]#<==> labels[topk_[i]],labels[topk_[j]] for topk_ = [i,j] in pytorch

    matches = (retrieved_labels == query_labels)#boolean matrix [M,K]
    hard = ~(retrieved_labels == query_labels).any(dim=1)#boolean vector [M], True if no match in top k

    return   (matches.any(dim=1).float().mean().item()   ,   hard.sum().item()   )
def mAP(S, labels):
    N = labels.shape[0]
    aps = []

    for i in range(N):
        # rank all images by similarity for query i
        ranking = torch.argsort(S[i], descending=True)

        # relevant = same-class images
        relevant = (labels[ranking] == labels[i]).float()                                  #[1,0,1,0]

        if relevant.sum() == 0:
            aps.append(0.0)
            continue

        # precision at each rank
        cum_relevant = torch.cumsum(relevant, dim=0)                                       #[1,1,2,2]
        precision_at_k = cum_relevant / torch.arange(1, len(relevant) + 1, device=S.device)#[1/1,1/2,2/3,2/4]
        #a/1,b/2,c/3,d/4,...

        # Average Precision for query i, keep only relevant ones
        ap = (precision_at_k * relevant).sum() / relevant.sum()                             #[1*1/1,0*1/2,1*2/3,0*2/4], added and divided by num relevant
        aps.append(ap.item())#done for one row

    return sum(aps) / len(aps)

for k in [1, 5, 10]:
    r_at_k = recallK(S, labels, k)[0]
    num_hard = recallK(S, labels, k)[1]

    print(f"Recall@{k}: {r_at_k*100:.2f}%, and hard queries: {num_hard}")

    for i in range(3):
        topk = torch.topk(S[i].cpu(), k=k)
        

        topk_scores = topk.values      # similarity values
        topk_indices = topk.indices  

        retrieved_imgs = [test_set.data[idx.item()] for idx in topk_indices]
        retrieved_labels = [test_set.targets[idx.item()] for idx in topk_indices]

        query_img = test_set.data[i]
        query_label = test_set.targets[i]

        plt.figure(figsize=(12,3))

        plt.subplot(1, k+1, 1)
        plt.imshow(query_img)
        plt.title("Query")
        plt.axis("off")

        for j in range(k):
            plt.subplot(1, k+1, j+2)
            plt.imshow(retrieved_imgs[j])
            plt.title(f"{retrieved_labels[j]}")
            plt.axis("off")

        plt.show()
        print(f"gate value of that image :{gate_val[i]}")
mAP_ = mAP(S, labels)
print(f"mAP: {mAP_*100}%")
"""