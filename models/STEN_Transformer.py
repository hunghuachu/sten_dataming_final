# STEN.py
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from utils.utility import get_sub_seqs


# --------------------------
# Utility: Positional Encoding
# --------------------------
def sinusoidal_position_encoding(seq_len, d_model, device=None, dtype=torch.float32):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = torch.from_numpy(pe).unsqueeze(0).to(dtype=dtype)
    if device is not None:
        pe = pe.to(device)
    return pe  # (1, seq_len, d_model)


# --------------------------
# InterLoss (JS divergence)
# --------------------------
class InterLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ['mean', 'none', 'sum']
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        y_pred: [B, C] (logits)
        y_true: [B, C] (labels or logits)
        compute JS divergence between softmax(y_pred) and softmax(y_true)
        """
        p = F.softmax(y_pred, dim=1).clamp(min=1e-12)
        q = F.softmax(y_true, dim=1).clamp(min=1e-12)
        m = 0.5 * (p + q)
        # KL(p||m) and KL(q||m)
        kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=1)
        kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=1)
        js = 0.5 * (kl_pm + kl_qm)
        if self.reduction == 'mean':
            return js.mean()
        elif self.reduction == 'sum':
            return js.sum()
        else:
            return js  # none


# --------------------------
# Transformer backbone
# --------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        out = self.layer(x)
        return self.norm(out)


class TransformerEncoderPool(nn.Module):
    """
    Projects input features to d_model, adds sinusoidal pos enc, runs TransformerEncoder,
    then mean-pools over time dimension and returns [B, d_model].
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=4*d_model,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
        self.register_buffer("_pe_dummy", torch.zeros(1))  # placeholder so state_dict consistent

    def forward(self, x):
        """
        x: [B, T, input_dim] (float)
        return: [B, d_model] pooled embedding
        """
        # ensure float32
        x = x.float()
        B, T, _ = x.shape
        x = self.proj(x)  # [B, T, d_model]
        pe = sinusoidal_position_encoding(T, self.d_model, device=x.device, dtype=x.dtype)
        x = x + pe  # broadcasting
        out = self.encoder(x)  # [B, T, d_model]
        pooled = out.mean(dim=1)
        return pooled  # [B, d_model]


# --------------------------
# Transformer ranking net (OTN)
# --------------------------
class TransformerRankNet(nn.Module):
    def __init__(self, input_dim, num_rank=10, d_model=256, nhead=8, num_layers=2, emb_dim=512, n_emb=128):
        super().__init__()
        self.num_rank = num_rank
        self.backbone = TransformerEncoderPool(input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model * num_rank, emb_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(emb_dim, n_emb)
        self.fc3 = nn.Linear(n_emb, num_rank)

    def forward(self, x1, x2):
        """
        x1: [B, num_rank, T, features]  # original sequences
        x2: [B, num_rank, T, features]  # shuffled/reordered sequences
        returns:
            logits: [B, num_rank]
            random_dis_embedding: [B, d_model * num_rank]
        """
        # ensure float
        x1 = x1.float()
        x2 = x2.float()

        B = x1.size(0)
        device = x1.device

        e1_list = []
        e2_list = []
        for i in range(self.num_rank):
            # process each sub-sequence with shared backbone
            seq_i_1 = x1[:, i, :, :]  # [B, T, features]
            seq_i_2 = x2[:, i, :, :]
            emb1 = self.backbone(seq_i_1)  # [B, d_model]
            emb2 = self.backbone(seq_i_2)
            e1_list.append(emb1)
            e2_list.append(emb2)

        # concat along feature axis
        e1 = torch.cat(e1_list, dim=1)  # [B, d_model*num_rank]
        e2 = torch.cat(e2_list, dim=1)

        logits = self.fc3(self.act(self.fc2(self.act(self.fc1(e2)))))
        return logits, e1  # logits for rank prediction; e1 used for distance


# --------------------------
# Transformer embedding net (DSN target)
# --------------------------
class TransformerEmbeddingNet(nn.Module):
    def __init__(self, input_dim, num_rank=10, d_model=256, nhead=8, num_layers=2, emb_dim=512):
        super().__init__()
        self.num_rank = num_rank
        self.backbone = TransformerEncoderPool(input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.fc = nn.Linear(d_model * num_rank, emb_dim)

    def forward(self, x):
        x = x.float()
        e_list = []
        for i in range(self.num_rank):
            emb = self.backbone(x[:, i, :, :])
            e_list.append(emb)
        e = torch.cat(e_list, dim=1)
        return self.fc(e)  # [B, emb_dim]


# --------------------------
# Datasets (kept behaviorally same as original)
# --------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs_lst, num_rank, seed=42):
        # seqs_lst: numpy array (n_segments, seq_len, features)
        # but original code expects each sample to be shape (num_rank * seq_len, features),
        # actually get_sub_seqs was called with seq_len * num_rank, so seqs are 2D segments.
        # To be compatible, we will reshape each segment into (num_rank, seq_len, features).
        self.num_rank = num_rank
        self.data = []
        for seg in seqs_lst:
            # seg shape (seq_len * num_rank, features) or (seq_len, features) depending on get_sub_seqs
            if seg.ndim == 2 and seg.shape[0] == num_rank * seg.shape[1] // num_rank:
                # ambiguous; but assume already (num_rank*seq_len, features)
                pass
            # try to reshape into (num_rank, seq_len, features)
            # if seg is (num_rank * seq_len, feat), we reshape:
            total_len = seg.shape[0]
            if total_len % num_rank == 0:
                seq_len = total_len // num_rank
                seg_rs = seg.reshape(num_rank, seq_len, seg.shape[1])
            else:
                # fallback: chunk equally (may truncate)
                seq_len = total_len // num_rank
                seg_rs = seg[:seq_len * num_rank].reshape(num_rank, seq_len, seg.shape[1])
            self.data.append(seg_rs.astype(np.float32))
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # return: data1 (num_rank, seq_len, feat), data2 (num_rank, seq_len, feat), label (num_rank,)
        data1 = self.data[idx]
        # pick a random other sample
        rand = np.random.randint(0, self.size - 1)
        while rand == idx:
            rand = np.random.randint(0, self.size - 1)
        data2 = self.data[rand]
        rank = np.arange(1, self.num_rank + 1)
        np.random.shuffle(rank)
        return data1, data2, rank  # all numpy arrays


class TestDataset(Dataset):
    def __init__(self, seqs_lst, num_rank, seed=42):
        self.num_rank = num_rank
        self.data = []
        for seg in seqs_lst:
            total_len = seg.shape[0]
            if total_len % num_rank == 0:
                seq_len = total_len // num_rank
                seg_rs = seg.reshape(num_rank, seq_len, seg.shape[1])
            else:
                seq_len = total_len // num_rank
                seg_rs = seg[:seq_len * num_rank].reshape(num_rank, seq_len, seg.shape[1])
            self.data.append(seg_rs.astype(np.float32))
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data1 = self.data[idx]
        arr = []
        while len(arr) < 5:
            num = np.random.randint(0, self.size - 1)
            if num != idx:
                arr.append(num)
        data2 = [self.data[a] for a in arr]
        rank = np.arange(1, self.num_rank + 1)
        np.random.shuffle(rank)
        return data1, data2, rank


# --------------------------
# STEN class
# --------------------------
class STEN:
    def __init__(self, seq_len_lst=None, seq_len=10, stride=1,
                 epoch=5, batch_size=256, lr=1e-5,
                 hidden_dim=256, rep_dim=100,
                 verbose=2, random_state=42,
                 random_strength=0.1, alpha=1, beta=1,
                 device='cuda'):

        if seq_len_lst is None:
            self.seq_len_lst = np.arange(10, 100, 10)
        else:
            self.seq_len_lst = seq_len_lst

        self.seq_len = seq_len
        self.num_rank = 10
        self.k = 5
        self.stride = stride

        self.epochs = epoch
        self.batch_size = batch_size
        self.lr = lr

        self.rep_dim = rep_dim
        self.hidden_dim = hidden_dim

        self.verbose = verbose
        self.random_state = random_state

        self.n_features = -1
        self.network = None
        self.Embnetwork = None

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.dl = 1e7
        self.alpha = alpha
        self.beta = beta

        self.epoch_steps = -1
        self.random_strength = random_strength

    def fit(self, x):
        """
        x: numpy array (n_samples, n_features)
        """
        self.n_features = x.shape[-1]
        seqs_lst = get_sub_seqs(x, seq_len=self.seq_len * self.num_rank, stride=self.stride)
        train_dataset = SeqDataset(seqs_lst, num_rank=self.num_rank, seed=self.random_state)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # build models
        self.network = TransformerRankNet(input_dim=self.n_features, num_rank=self.num_rank,
                                          d_model=self.hidden_dim).to(self.device).float()
        self.Embnetwork = TransformerEmbeddingNet(input_dim=self.n_features, num_rank=self.num_rank,
                                                  d_model=self.hidden_dim).to(self.device).float()

        random_criterion = torch.nn.MSELoss(reduction='mean')
        criterion = InterLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=1e-5)

        self.network.train()
        for epoch in range(self.epochs):
            t0 = time.time()
            losses = []
            with tqdm(total=len(train_loader), desc=f'epoch {epoch+1}/{self.epochs}', ncols=140) as pbar:
                for idx, (batch_data1, batch_data2, batch_label) in enumerate(train_loader):
                    # batch_data1, batch_data2: list -> numpy arrays from dataset
                    # convert to tensors
                    # batch_data1 shape (B, num_rank, seq_len, feat) in numpy
                    batch_size = batch_data1.shape[0]

                    seq_batch_data = torch.tensor(batch_data1, dtype=torch.float32, device=self.device)  # [B, R, T, F]
                    seq_batch_data_random = torch.tensor(batch_data2, dtype=torch.float32, device=self.device)

                    # batch_label is numpy array of shape [B, num_rank]
                    batch_label = torch.tensor(batch_label, dtype=torch.long, device=self.device)  # [B, R] values 1..R

                    # construct shuffled/reordered sequences per-sample (same as original code)
                    seq_batch_shuffle_list = []
                    seq_batch_shuffle_list_random = []
                    for n in range(batch_size):
                        shuffle_idx = (batch_label[n] - 1).long()  # indices 0..R-1, shape [R]
                        seq_batch_shuffle = seq_batch_data[n, shuffle_idx, :, :]  # [R, T, F]
                        seq_batch_shuffle_list.append(seq_batch_shuffle)
                        seq_batch_shuffle_rand = seq_batch_data_random[n, shuffle_idx, :, :]
                        seq_batch_shuffle_list_random.append(seq_batch_shuffle_rand)

                    seq_batch_shuffle_data = torch.stack(seq_batch_shuffle_list, dim=0).to(self.device)  # [B, R, T, F]
                    seq_batch_shuffle_data_random = torch.stack(seq_batch_shuffle_list_random, dim=0).to(self.device)

                    # forward
                    pred, pred_dis = self.network(seq_batch_data, seq_batch_shuffle_data)
                    _, pred_dis_random = self.network(seq_batch_data_random, seq_batch_shuffle_data_random)

                    # detach predicted distance embeddings
                    pred_dis = pred_dis.detach()
                    pred_dis_random = pred_dis_random.detach()

                    # normalized similarity (L1 normalization then dot)
                    xy_dis = (F.normalize(pred_dis, p=1, dim=1) * F.normalize(pred_dis_random, p=1, dim=1)).sum(dim=1)

                    # target embedding from Embnetwork (detach)
                    pred_target = self.Embnetwork(seq_batch_data).detach()
                    pred_target_random = self.Embnetwork(seq_batch_data_random).detach()
                    x_y_dis = (F.normalize(pred_target, p=1, dim=1) * F.normalize(pred_target_random, p=1, dim=1)).sum(dim=1)

                    # rank loss
                    # original code used seq_batch_label as float (preserve behavior: feed permutation vector as 'true' distribution)
                    seq_batch_label = batch_label.float().to(self.device)  # shape [B, R] but currently [B, R] ; original expected float
                    # however InterLoss expects [B, C] tensors. We need logits shaped [B, R] for pred and [B, R] for seq_batch_label
                    loss_rank = criterion(pred, seq_batch_label)

                    # distance loss
                    dis_loss_raw = random_criterion(xy_dis, x_y_dis)
                    dis_loss = self.dl * dis_loss_raw

                    loss = loss_rank + self.alpha * dis_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    pbar.set_postfix(loss=f"{np.mean(losses):.6f}")
                    pbar.update(1)

                    if self.epoch_steps != -1 and idx > self.epoch_steps:
                        break

            print(f"epoch {epoch+1}/{self.epochs}: loss={np.mean(losses):.6f} time={(time.time()-t0):.1f}s")

        return

    def decision_function(self, x):
        length = len(x)
        seqs = get_sub_seqs(x, seq_len=self.seq_len * self.num_rank, stride=1)
        test_dataset = TestDataset(seqs, num_rank=self.num_rank)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        random_criterion = torch.nn.MSELoss(reduction='none')
        criterion = InterLoss(reduction='none')

        self.network.eval()
        self.Embnetwork.eval()

        score_lst = []
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f'testing len={self.seq_len}', ncols=140) as pbar:
                for batch_data1, batch_data2, batch_label in test_loader:
                    batch_size = batch_data1.shape[0]
                    seq_batch_data = torch.tensor(batch_data1, dtype=torch.float32, device=self.device)
                    # build shuffled per-sample
                    batch_label_t = torch.tensor(batch_label, dtype=torch.long, device=self.device)
                    seq_batch_shuffle_list = []
                    for n in range(batch_size):
                        idxs = (batch_label_t[n] - 1).long()
                        seq_batch_shuffle_list.append(seq_batch_data[n, idxs, :, :])
                    seq_batch_shuffle_data = torch.stack(seq_batch_shuffle_list, dim=0).to(self.device)

                    dis_loss_arr = []
                    rank_loss_arr = []
                    for a in range(5):
                        seq_batch_data_random = torch.tensor(batch_data2[a], dtype=torch.float32, device=self.device)
                        # reorder random by same permutation
                        seq_batch_shuffle_list_random = []
                        for r in range(batch_size):
                            idxs = (batch_label_t[r] - 1).long()
                            seq_batch_shuffle_list_random.append(seq_batch_data_random[r, idxs, :, :])
                        seq_batch_shuffle_data_random = torch.stack(seq_batch_shuffle_list_random, dim=0).to(self.device)

                        # forward
                        pred, pred_dis = self.network(seq_batch_data, seq_batch_shuffle_data)
                        _, pred_dis_random = self.network(seq_batch_data_random, seq_batch_shuffle_data_random)

                        pred_dis = pred_dis.detach()
                        pred_dis_random = pred_dis_random.detach()

                        xy_dis = (F.normalize(pred_dis, p=1, dim=1) * F.normalize(pred_dis_random, p=1, dim=1)).sum(dim=1)
                        pred_target = self.Embnetwork(seq_batch_data).detach()
                        pred_target_random = self.Embnetwork(seq_batch_data_random).detach()
                        x_y_dis = (F.normalize(pred_target, p=1, dim=1) * F.normalize(pred_target_random, p=1, dim=1)).sum(dim=1)

                        dis_loss = random_criterion(xy_dis, x_y_dis).cpu().numpy()
                        dis_loss_arr.append(dis_loss)

                        # rank loss
                        # prepare label as float distribution (same as training)
                        seq_batch_label_float = batch_label_t.float().to(self.device)
                        pred_s = F.softmax(pred, dim=1)
                        label_s = F.softmax(seq_batch_label_float, dim=1)
                        item_loss = torch.abs(pred_s - label_s).cpu().numpy()

                        rank_loss = criterion(pred, seq_batch_label_float).flatten().cpu().numpy()
                        # reshape and compute item/reshape normalization (matching original code)
                        reshape_loss = np.zeros((batch_size, self.num_rank))
                        reshape_loss[:] = rank_loss[:, np.newaxis]
                        rank_item = item_loss / reshape_loss
                        rank_loss_arr.append(rank_item)

                    dis_loss = np.average(dis_loss_arr, axis=0)
                    dis_loss_reshape = np.zeros((batch_size, self.num_rank))
                    dis_loss_reshape[:] = dis_loss[:, np.newaxis]
                    rank_loss = np.average(rank_loss_arr, axis=0)

                    anomaly_score = rank_loss + self.beta * dis_loss_reshape
                    score_lst.append(anomaly_score)
                    pbar.update(1)

        score_arr = np.concatenate(score_lst, axis=0)
        scores = self.get_score(score_arr, length)
        _max_, _min_ = np.max(scores), np.min(scores)
        if _max_ - _min_ > 0:
            norm_scores = (scores - _min_) / (_max_ - _min_)
        else:
            norm_scores = scores
        padding = np.zeros(self.seq_len - 1)
        assert padding.shape[0] + norm_scores.shape[0] == x.shape[0]
        return np.hstack((padding, norm_scores))

    def get_score(self, input_array, length):
        # input_array: (n_segments, num_rank)
        S_lst = [[] for _ in range(length)]
        for i in range(len(input_array)):
            for j in range(self.num_rank):
                idx = i + (j + 1) * self.seq_len - 1
                if idx < length:
                    S_lst[idx].append(input_array[i][j])
        New_lst = [lst for lst in S_lst if lst]
        avg_lst = []
        for seqs in New_lst:
            avg_lst.append(sum(seqs) / len(seqs))
        return np.array(avg_lst)


