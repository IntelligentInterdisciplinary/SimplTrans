import torch
from torch.nn import functional as F
from torchvision import transforms, datasets
from loguru import logger
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter

class Config:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset="mnist"#mnist,cifar10
    epoch = 100
    batchsize = 256
    num_workers = 8
    torch.manual_seed(42)
    log_name = "experiment.log"
    best_acc = 0
    best_epoch = 0
    alpha = 0.001
    mode="test"#random,diagonal,identity,test,vit
    image_size=28
    patch_size=4
    num_classes=10
    dim=48
    depth=12
    heads=12
    mlp_dim=192
    pool="cls"
    channels=1
    dim_head=4
    dropout=0.1
    emb_dropout=0.1
config = Config()

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class ModAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.dim=dim
        self.heads=heads
        if config.mode=="random":
            self.A=Parameter(torch.randn(int(dim/heads), int(dim/heads), device=config.device))
        elif config.mode=="diagonal":
            self.A=Parameter(torch.randn(1, int(dim/heads), device=config.device))
        elif config.mode=="identity":
            pass
        elif config.mode=="test":
            pass
        self.W=torch.nn.Linear(dim,dim,bias=False)
        self.dropout1=torch.nn.Dropout(dropout)
        self.dropout2=torch.nn.Dropout(dropout)
    def forward(self, x):
        xx=rearrange(x, "b n (h d) -> b h n d", h=self.heads)
        if config.mode=="random":
            xx=torch.matmul(xx,self.A)
            xx=torch.matmul(xx,xx.transpose(-1,-2))
        elif config.mode=="diagonal":
            xx=torch.mul(xx,self.A)
            xx=torch.matmul(xx,xx.transpose(-1,-2))
        elif config.mode=="identity":
            xx=torch.matmul(xx,xx.transpose(-1,-2))
        elif config.mode=="test":
            y=self.W(x)
            a=F.layer_norm(self.dropout2(x+y),(self.dim,))
            return a
        xx=self.dropout1(xx)
        v=self.W(x)
        v=rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        y=torch.matmul(xx,v)
        y=rearrange(y, "b h n d -> b n (h d)")
        a=F.layer_norm(self.dropout2(x+y),(self.dim,))
        return a


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ModTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            ModAttention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        pool=config.pool,
        channels=config.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class ModViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        pool=config.pool,
        channels=config.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ModTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


train_transform = transforms.Compose([
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])
if config.dataset=="mnist":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/root/workspace/mnist', train=True, transform=train_transform, download=True),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/root/workspace/mnist', train=False, transform=test_transform, download=True),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.num_workers,
    )
elif config.dataset=="cifar10":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/root/workspace/cifar10', train=True, transform=train_transform, download=True),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/root/workspace/cifar10', train=False, transform=test_transform, download=True),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.num_workers,
    )


if __name__ == "__main__":
    logger.add(config.log_name)
    if config.mode=='vit':
        net=ViT().to(config.device)
    else:
        net=ModViT().to(config.device)
    criterion=torch.nn.CrossEntropyLoss(size_average=True).to(config.device)
    optimizer=torch.optim.Adam(net.parameters(),lr=config.alpha)
    scheduler=lr_scheduler.CosineAnnealingLR(optimizer,config.epoch)
    for epoch_id in range(config.epoch):
        # train
        net.train()
        with tqdm(len(train_loader)) as pbar:
            for batch_id, sample in enumerate(train_loader):
                optimizer.zero_grad()
                data, label= sample[0].to(config.device), sample[1].to(config.device)
                z = net(data)
                cost=criterion(z,label)
                cost.backward()
                optimizer.step()
                batch_acc=(z.argmax(axis=1) == label).sum()/label.shape[0]
                pbar.update(1)
                pbar.set_description("Epoch: {}, Batch: {}/{}, Train Acc: {:.5f}".format(epoch_id, batch_id, len(train_loader), batch_acc))
        #test
        with torch.no_grad():
            net.eval()
            total, correct = 0.0, 0.0
            for batch_id, sample in enumerate(test_loader):
                data, label = sample[0].to(config.device), sample[1]          
                a_pred = net(data).cpu().argmax(dim=1)
                a_true = label
                correct += (a_pred == a_true).sum().item()
                total += a_true.shape[0]
            scheduler.step()
            logger.info(f"Epoch: {epoch_id} Test acc={correct/total:.4f}")
            if config.best_acc<correct/total:
                config.best_acc=correct/total
                config.best_epoch=epoch_id
            logger.info(f"Best Epoch: {config.best_epoch} Best acc={config.best_acc:.4f}")