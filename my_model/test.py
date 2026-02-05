from model.ONETRANS import ONETRANS
import torch

bsize = 2
num_layers = 5
max_seq_len = [128, 64, 32, 16, 8]
ns_seq_len = 2
d_model = 64
num_heads = 2
ffn_layer_hidden_dim = 128
main_tower_units = [128, 2]
x = torch.rand(bsize, max_seq_len[0]+ns_seq_len, d_model)
in_seq_len = torch.tensor([19, 128])

model = ONETRANS(
    num_layers=num_layers,
    max_seq_len=max_seq_len,
    ns_seq_len=ns_seq_len,
    d_model=d_model,
    num_heads=num_heads,
    ffn_layer_hidden_dim=ffn_layer_hidden_dim,
    main_tower_units=main_tower_units
)
ret = model(x, in_seq_len)
print(ret.shape)