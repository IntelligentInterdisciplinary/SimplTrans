import torch

model="mtransformer"#"transformer","mtransformer"
n_layers = 6
d_model = 512
d_ff = 2048
n_heads = 8
d_k = 64
d_v = 64
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 32
epoch_num = 100
lr = 3e-4

max_len = 60
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = True
# NoamOpt
use_noamopt = True
gpu='2, 3'
seed=42

data_dir = './data'
train_data_path = './data/json/train.json'
dev_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'

model_path = f'./experiment/{model}.pth'
log_path = f'./experiment/{model}.log'
output_path = f'./experiment/{model}.txt'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0, 1]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')