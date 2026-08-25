import sys
from pathlib import Path
from matplotlib.pyplot import step
from matplotlib.transforms import Transform
import torch
sys.path.append(str(Path(__file__).parent.parent))
from eecs498.utils import attention_visualizer
from a5_helper import get_toy_data, inference,draw
from a5_helper import train as train_transformer
from a5_helper import val as val_transformer
import os
from sklearn.model_selection import train_test_split
from ImageCaption.models.transformers import (
    generate_token_dict,
    CrossEntropyLoss,
    position_encoding_simple,
    position_encoding_sinusoid,
    AddSubDataset,
    LabelSmoothingLoss,CrossEntropyLoss,
    Transformer
)

SPECIAL_TOKENS = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + SPECIAL_TOKENS

convert_str_to_tokens = generate_token_dict(vocab)
data = get_toy_data(os.path.join("./datasets", "two_digit_op.json"))
BATCH_SIZE = 16

X, y = data["inp_expression"], data["out_expression"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

inp_seq_len = 9
out_seq_len = 5
BATCH_SIZE = 256

# You should change these!
num_heads = 4
emb_dim = 64
dim_feedforward = 16
dropout = 0.2
num_enc_layers = 4
num_dec_layers = 4
vocab_len = len(vocab)
loss_func = CrossEntropyLoss
poss_enc = position_encoding_simple
num_epochs = 200
warmup_interval = None
lr = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_data = AddSubDataset(
    X_train,
    y_train,
    convert_str_to_tokens,
    SPECIAL_TOKENS,
    emb_dim,
    position_encoding_sinusoid,
)
valid_data = AddSubDataset(
    X_test,
    y_test,
    convert_str_to_tokens,
    SPECIAL_TOKENS,
    emb_dim,
    position_encoding_sinusoid,
)


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)

model  = Transformer(
    num_heads,
    emb_dim,
    dim_feedforward,
    dropout,
    num_enc_layers,
    num_enc_layers,
    vocab_len
)

# Small_Data Train

small_dataset = torch.utils.data.Subset(
    train_data, torch.linspace(0, len(train_data) - 1, steps=4).long()
)
small_train_loader = torch.utils.data.DataLoader(
    small_dataset, batch_size=4,shuffle=False
)

train_small_model = train_transformer(
    model,
    small_train_loader,
    small_train_loader,
    loss_func,
    num_epochs,
    lr=lr,
    batch_size=BATCH_SIZE,
    warmup_interval=warmup_interval,
    device=DEVICE
)
# Overfitted accuracy
print(
    "Overfitted accuracy: ",
    "{:.4f}".format(
        val_transformer(
            train_small_model,
            small_train_loader,
            CrossEntropyLoss,
            batch_size=4,
            device=DEVICE,
        )[1]
    ),
)

# Training the model with complete data
trained_model = train_transformer(
    model,
    train_loader,
    valid_loader,
    loss_func,
    num_epochs,
    lr=lr,
    batch_size=BATCH_SIZE,
    warmup_interval=warmup_interval,
    device=DEVICE
)

weights_path = os.path.join("./weights", "transformer.pt")
torch.save(trained_model.state_dict(), weights_path)

# Final validation accuracy
print(
    "Final Model accuracy: ",
    "{:.4f}".format(
        val_transformer(
            trained_model, valid_loader, LabelSmoothingLoss, 4, device=DEVICE
        )[1]
    ),
)


for it in valid_loader:
    it
    break
inp, inp_pos, out, out_pos = it
opposite_tokens_to_str = {v: k for k, v in convert_str_to_tokens.items()}
device = torch.device("cpu")
model = model.to(device)
inp_pos = inp_pos.to(device)
out_pos = out_pos.to(device)
out = out.to(device)
inp = inp.to(device)

inp_exp = inp[:1, :]
inp_exp_pos = inp_pos[:1]
out_pos_exp = out_pos[:1, :]
inp_seq = [opposite_tokens_to_str[w.item()] for w in inp_exp[0]]
print(
    "Input sequence: \n",
    inp_seq[0]
    + " "
    + inp_seq[1]
    + " "
    + inp_seq[2]
    + inp_seq[3]
    + " "
    + inp_seq[4]
    + " "
    + inp_seq[5]
    + " "
    + inp_seq[6]
    + inp_seq[7]
    + " "
    + inp_seq[8],
)

out_seq_ans, _ = inference(
    trained_model, inp_exp, inp_exp_pos, out_pos_exp, out_seq_len
)

trained_model.eval()

print("Output Sequence:", end="\t")
res = "BOS "
for i in range(1, out_seq_ans.size(1)):
    sym = opposite_tokens_to_str[out_seq_ans[0, i].item()]
    if sym == "EOS":
        break
    res += sym + " "
print(res)
