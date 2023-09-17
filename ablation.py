import torch
from torch.nn import functional as F
from torchvision import transforms, datasets
from torchdiffeq import odeint
from torch.utils.data import DataLoader
from loguru import logger
from torch.nn.parameter import Parameter
from tqdm import tqdm

class Config:
    epoch = 1000
    batchsize = 1024
    num_workers = 8
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(22)
    logger_name="base"
    best_epoch=0
    best_acc=0
    xsize=784
    asize=10
    alpha=0.001
    block_num=3
    mode='mTransformer'
config=Config()

train_transforms = transforms.Compose([
        transforms.ToTensor(),
])
test_transforms = transforms.Compose([
        transforms.ToTensor(),
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("mnist", train=True, transform=train_transforms, download=True),
    batch_size=config.batchsize,
    shuffle=True,
    num_workers=config.num_workers,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("mnist", train=False, transform=test_transforms, download=True),
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=config.num_workers,
)

class mBlock(torch.nn.Module):
    def __init__(self):
        super(mBlock,self).__init__()
        self.WV=Parameter(torch.zeros(config.xsize, config.xsize, device=config.device))
        self.norm=torch.nn.LayerNorm(config.xsize)
        torch.nn.init.kaiming_uniform_(self.WV, mode='fan_out', nonlinearity='relu')
    def forward(self,x):
        x=x.view(-1,28,28)
        QK=torch.bmm(x,x.transpose(1,2)).view(-1,784)
        x=x.view(-1,784)
        V=torch.matmul(x,self.WV)
        a=self.norm(x+torch.mul(QK,V))
        return a

class mTransformer(torch.nn.Module):
    def __init__(self):
        super(mTransformer,self).__init__()
        self.blocks=torch.nn.ModuleList([mBlock() for _ in range(config.block_num)])
        self.W=Parameter(torch.zeros(config.asize, config.xsize, device=config.device))
    def forward(self,data):
        batch_size = data.shape[0]
        x = data.to(config.device).view(batch_size, -1)
        for block in self.blocks:
            x=block(x)
        a=torch.matmul(x,self.W.T)
        return a

class Block(torch.nn.Module):
    def __init__(self):
        super(Block,self).__init__()
        self.WQ=Parameter(torch.zeros(config.xsize, config.xsize, device=config.device))
        self.WK=Parameter(torch.zeros(config.xsize, config.xsize, device=config.device))
        self.WV=Parameter(torch.zeros(config.xsize, config.xsize, device=config.device))
        self.norm=torch.nn.LayerNorm(config.xsize)
        torch.nn.init.kaiming_uniform_(self.WQ, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.WK, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.WV, mode='fan_out', nonlinearity='relu')
    def forward(self,x):
        Q=torch.matmul(x,self.WQ).view(-1,28,28)
        K=torch.matmul(x,self.WK).view(-1,28,28)
        QK=torch.softmax((torch.bmm(Q,K.transpose(1,2))/28).view(-1,784),dim=-1)
        V=torch.matmul(x,self.WV)
        a=self.norm(x+torch.mul(QK,V))
        return a

class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.blocks=torch.nn.ModuleList([Block() for _ in range(config.block_num)])
        self.W=Parameter(torch.zeros(config.asize, config.xsize, device=config.device))
    def forward(self,data):
        batch_size = data.shape[0]
        x = data.to(config.device).view(batch_size, -1)
        for block in self.blocks:
            x=block(x)
        a=torch.matmul(x,self.W.T)
        return a

if __name__ == "__main__":
    if config.mode=="mTransformer":
        net = mTransformer().to(config.device)
    else:
        net = Transformer().to(config.device)
    criterion=torch.nn.CrossEntropyLoss(size_average=True).to(config.device)
    optimizer=torch.optim.Adam(net.parameters(),lr=config.alpha)
    
    for epoch_id in range(config.epoch):
        net.train()
        with tqdm(len(train_loader)) as pbar:
            for batch_id, (data,label) in enumerate(train_loader):
                optimizer.zero_grad()
                data, label= data.to(config.device), label.to(config.device)
                zt = net(data)
                cost=criterion(zt,label)
                cost.backward()
                optimizer.step()
                batch_acc=(zt.argmax(axis=1) == label).sum()/label.shape[0]
                pbar.update(1)
                pbar.set_description("Epoch: {}, Batch: {}/{}, Train Acc: {:.5f}".format(epoch_id, batch_id, len(train_loader), batch_acc))

        with torch.no_grad():
            net.eval()
            total, correct = 0.0, 0.0
            for batch_id, (data,label) in enumerate(test_loader):
                data = data.to(config.device)            
                a_pred = net(data).cpu().argmax(dim=1)
                a_true = label
                correct += (a_pred == a_true).sum().item()
                total += a_true.shape[0]
            logger.info(f"Epoch: {epoch_id} Test acc={correct/total:.4f}")
            if config.best_acc<correct/total:
                config.best_acc=correct/total
                config.best_epoch=epoch_id
            logger.info(f"Best Epoch: {config.best_epoch} Best acc={config.best_acc:.4f}")
