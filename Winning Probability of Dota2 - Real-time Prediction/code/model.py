import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Model']


def attend(q, k, v, *, mask=None):
    # q: Float[bs, n_q, d_k]
    # k: Float[bs, n_e, d_k]
    # v: Float[bs, n_e, d_v]
    a = torch.einsum("bik,bjk->bij", q, k) / (q.size(2) ** 0.5)
    if mask is not None:
        a = a.masked_fill(1 - mask, -np.inf)
    a = F.softmax(a, dim=2)  # a: Float[bs, n_q, n_e]
    r = torch.einsum("bik,bkj->bij", a, v)  # r: Float[bs, n_q, d_v]
    return r, a


class PositionEncoding(nn.Module):
    def __init__(self, d):
        super(PositionEncoding, self).__init__()
        self.d = d
        assert d % 2 == 0
        self.coef = nn.Parameter(1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float) / d), requires_grad=False)

    def forward(self, x):
        x = x.unsqueeze(-1)
        s = list(x.shape)
        s[-1] = self.d // 2
        x = x.expand(*s).to(torch.float)
        return torch.cat([torch.sin(x * self.coef), torch.cos(x * self.coef)], dim=-1)


class PointwiseFeedForward(nn.Module):
    def __init__(self, d, d_hid, *, drop_prob=0.0):
        super(PointwiseFeedForward, self).__init__()
        self.d = d
        self.w1 = nn.Linear(d, d_hid)
        self.w2 = nn.Linear(d_hid, d)
        self.ln = nn.LayerNorm(d)
        if drop_prob > 0.0:
            self.dropout = nn.Dropout(drop_prob)
        else:
            self.dropout = None
        nn.init.kaiming_normal_(self.w1.weight, nonlinearity='relu')
        nn.init.zeros_(self.w1.bias)
        nn.init.kaiming_normal_(self.w2.weight, nonlinearity='linear')
        nn.init.zeros_(self.w2.bias)

    def forward(self, x):
        # x: Float[..., d]
        s = x.shape
        x = x.view(-1, self.d)
        res = x
        x = self.w2(F.relu(self.w1(x)))
        if self.dropout:
            x = self.dropout(x)
        # return: Float[..., d]
        return self.ln(x + res).view(*s)


class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_h, *, drop_prob=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d = d
        self.n_h = n_h
        self.d_k = d // n_h
        self.q_l = nn.Linear(d, d)
        self.k_l = nn.Linear(d, d)
        self.v_l = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.o_l = nn.Linear(d, d)
        if drop_prob > 0.0:
            self.dropout = nn.Dropout(drop_prob)
        else:
            self.dropout = None
        nn.init.kaiming_normal_(self.q_l.weight, nonlinearity='linear')
        nn.init.zeros_(self.q_l.bias)
        nn.init.kaiming_normal_(self.k_l.weight, nonlinearity='linear')
        nn.init.zeros_(self.k_l.bias)
        nn.init.kaiming_normal_(self.v_l.weight, nonlinearity='linear')
        nn.init.zeros_(self.v_l.bias)
        nn.init.kaiming_normal_(self.o_l.weight, nonlinearity='linear')
        nn.init.zeros_(self.o_l.bias)

    def forward(self, q, k, v, *, mask=None):
        # q: Float[bs, n_q, d]
        # k: Float[bs, n_e, d]
        # v: Float[bs, n_e, d]
        # mask: Byte[bs, n_q, n_e]
        bs, n_q, _ = q.shape
        _, n_e, _ = k.shape
        res = q
        q = self.q_l(q).view(bs, n_q, self.n_h, self.d_k)
        k = self.k_l(k).view(bs, n_e, self.n_h, self.d_k)
        v = self.v_l(v).view(bs, n_e, self.n_h, self.d_k)
        q = q.permute(0, 2, 1, 3).contiguous().view(bs * self.n_h, n_q, self.d_k)
        k = k.permute(0, 2, 1, 3).contiguous().view(bs * self.n_h, n_e, self.d_k)
        v = v.permute(0, 2, 1, 3).contiguous().view(bs * self.n_h, n_e, self.d_k)
        if mask is not None:
            mask = mask.view(bs, 1, n_q, n_e)
            mask = mask.repeat(1, self.n_h, 1, 1)
            mask = mask.view(bs * self.n_h, n_q, n_e)
        x, a = attend(q, k, v, mask=mask)
        x = x.view(bs, self.n_h, n_q, self.d_k)
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, n_q, self.d)
        x = self.o_l(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.ln(x + res)
        a = a.view(bs, self.n_h, n_q, n_e).mean(dim=1)
        # returns: Float[bs, n_q, d], Float[bs, n_q, n_e]
        return x, a


class EncoderLayer(nn.Module):
    def __init__(self, d, n_h, d_hid, *, drop_prob=0.0):
        super(EncoderLayer, self).__init__()
        self.d = d
        self.n_h = n_h
        self.d_hid = d_hid
        self.s_s_att = MultiHeadAttention(d, n_h, drop_prob=drop_prob)
        self.s_e_att = MultiHeadAttention(d, n_h, drop_prob=drop_prob)
        self.e_s_att = MultiHeadAttention(d, n_h, drop_prob=drop_prob)
        self.e_e_att = MultiHeadAttention(d, n_h, drop_prob=drop_prob)
        self.s_ff = PointwiseFeedForward(d, d_hid, drop_prob=drop_prob)
        self.e_ff = PointwiseFeedForward(d, d_hid, drop_prob=drop_prob)

    def forward(self, s, e, s_e_mask, e_s_mask, e_e_mask):
        # s: Float[bs, max_t, 10, d]
        # e: Float[bs, max_e, d]
        # s_e_mask: Byte[bs, max_t, 10, max_e]
        # e_s_mask: Byte[bs, max_e, max_t, 10]
        # e_e_mask: Byte[bs, max_e, max_e]
        bs, max_t, _, _ = s.shape
        _, max_e, _ = e.shape
        s = s.view(bs * max_t, 10, self.d)
        s_s, _ = self.s_s_att(s, s, s)
        s = s.view(bs, max_t * 10, self.d)
        s_e, _ = self.s_e_att(s, e, e, mask=s_e_mask.view(bs, max_t * 10, max_e))
        e_s, _ = self.e_s_att(e, s, s, mask=e_s_mask.view(bs, max_e, max_t * 10))
        e_e, _ = self.e_e_att(e, e, e, mask=e_e_mask)
        s = s_s.view(bs, max_t, 10, self.d) + s_e.view(bs, max_t, 10, self.d)
        e = e_s + e_e
        s = self.s_ff(s)
        e = self.e_ff(e)
        return s, e


class IndividualExtractor(nn.Module):
    def __init__(self, d_si, d, d_hid, *, drop_prob=0.0):
        super(IndividualExtractor, self).__init__()
        self.d_si = d_si
        self.d = d
        self.bn = nn.BatchNorm1d(d_si, affine=False)
        self.l = nn.Linear(d_si, d)
        self.hero_emb = nn.Embedding(121, d)
        self.team_emb = nn.Embedding(2, d)
        self.ff = PointwiseFeedForward(d, d_hid, drop_prob=drop_prob)
        nn.init.kaiming_normal_(self.l.weight, nonlinearity='linear')
        nn.init.zeros_(self.l.bias)

    def forward(self, s, hero):
        # s: Float[bs, max_t, 10, d_si]
        # hero: Long[bs, 10]
        bs, max_t, _, _ = s.shape
        s = s.view(bs * max_t * 10, self.d_si)
        s = self.bn(s)
        s = self.l(s)
        s = s.view(bs, max_t, 10, self.d)
        hero = self.hero_emb(hero)
        s = s + hero.view(bs, 1, 10, self.d).expand_as(s)
        team = self.team_emb(torch.LongTensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).to(self.team_emb.weight.device))
        s = s + team.view(1, 1, 10, self.d).expand_as(s)
        s = self.ff(s)
        # return: Float[bs, max_t, 10, d]
        return s


class GlobalExtractor(nn.Module):
    def __init__(self, d_sg, d, d_hid, *, drop_prob=0.0):
        super(GlobalExtractor, self).__init__()
        self.d_sg = d_sg
        self.d = d
        self.bn = nn.BatchNorm1d(d_sg, affine=False)
        self.l = nn.Linear(d_sg, d)
        self.ff = PointwiseFeedForward(d, d_hid, drop_prob=drop_prob)
        nn.init.kaiming_normal_(self.l.weight, nonlinearity='linear')
        nn.init.zeros_(self.l.bias)

    def forward(self, s):
        # s: Float[bs, max_t, d_sg]
        bs, max_t, _ = s.shape
        s = s.view(bs * max_t, self.d_sg)
        s = self.bn(s)
        s = self.l(s)
        s = s.view(bs, max_t, self.d)
        s = self.ff(s)
        # return: Float[bs, max_t, d]
        return s


class Model(nn.Module):
    def __init__(self, d_si, d_sg, d, d_hid, n_h, n_layers, *, drop_prob=0.0):
        super(Model, self).__init__()
        self.d = d
        self.ie = IndividualExtractor(d_si, d, d_hid, drop_prob=drop_prob)
        self.ge = GlobalExtractor(d_sg, d, d_hid, drop_prob=drop_prob)
        self.event_emb = nn.Embedding(59, d)
        self.enc = nn.ModuleList([EncoderLayer(d, n_h, d_hid, drop_prob=drop_prob) for _ in range(n_layers)])
        self.att_si = MultiHeadAttention(d, n_h, drop_prob=drop_prob)
        self.att_e = MultiHeadAttention(d, n_h, drop_prob=drop_prob)
        self.classifier = nn.Linear(d, 1)
        self.pos_encoder = PositionEncoding(d)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def get_mask_g_e(bs, max_t, max_e, e_t, e_len, *, device):
        i, j, k = torch.meshgrid(torch.arange(bs, device=device),
                                 torch.arange(max_t, device=device),
                                 torch.arange(max_e, device=device))
        mask = (k < e_len[i]) & (e_t[i, k] < j * 60)
        return mask

    @staticmethod
    def get_mask_s_e(bs, max_t, max_e, e_t, e_mask, e_len, *, device):
        i, j, k, l = torch.meshgrid(torch.arange(bs, device=device),
                                    torch.arange(max_t, device=device),
                                    torch.arange(10, device=device),
                                    torch.arange(max_e, device=device))
        mask = (l < e_len[i]) & (e_t[i, l] < j * 60) & e_mask[i, l, k]
        return mask

    @staticmethod
    def get_mask_e_s(bs, max_t, max_e, e_t, e_mask, e_len, *, device):
        i, j, k, l = torch.meshgrid(torch.arange(bs, device=device),
                                    torch.arange(max_e, device=device),
                                    torch.arange(max_t, device=device),
                                    torch.arange(10, device=device))
        mask = ((k == (e_t[i, j].clamp(min=0) // 60)) & e_mask[i, j, l]) | (j >= e_len[i])
        return mask

    @staticmethod
    def get_mask_e_e(bs, max_e, e_t, e_len, *, device):
        i, j, k = torch.meshgrid(torch.arange(bs, device=device),
                                 torch.arange(max_e, device=device),
                                 torch.arange(max_e, device=device))
        mask = (k < e_len[i]) & (e_t[i, j] >= e_t[i, k])
        return mask

    def forward(self, si, hero, sg, e, e_t, e_mask, e_len):
        # si: Float[bs, max_t, 10, d_si]
        # hero: Long[bs, 10]
        # sg: Float[bs, max_t, d_sg]
        # e: Long[bs, max_e]
        # e_t: Long[bs, max_e]
        # e_mask: Byte[bs, max_e, 10]
        # e_len: Long[bs]
        bs, max_t, _, _ = si.shape
        _, max_e = e.shape
        pos_enc = self.pos_encoder(torch.arange(max_t).to(si.device))  # pos_enc: Float[max_t, d]

        si = self.ie(si, hero)
        si = si + pos_enc.view(1, max_t, 1, self.d).expand_as(si)
        si = si * torch.FloatTensor([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]).to(si.device).view(1, 1, 10, 1).expand_as(si)

        sg = self.ge(sg)
        sg = sg + pos_enc.view(1, max_t, self.d).expand_as(sg)

        e = self.event_emb(e)  # e: Float[bs, max_e, d]
        e = e + self.pos_encoder(e_t.to(torch.float) / 60.0)

        mask_s_e = Model.get_mask_s_e(bs, max_t, max_e, e_t, e_mask, e_len, device=si.device)
        mask_e_s = Model.get_mask_e_s(bs, max_t, max_e, e_t, e_mask, e_len, device=si.device)
        mask_e_e = Model.get_mask_e_e(bs, max_e, e_t, e_len, device=si.device)
        for i in range(len(self.enc)):
            si, e = self.enc[i](si, e, mask_s_e, mask_e_s, mask_e_e)

        si = si.view(bs * max_t, 10, self.d)
        sg = sg.view(bs * max_t, 1, self.d)
        x_si, a_si = self.att_si(sg, si, si)  # x_si: Float[bs * max_t, 1, d]
        x_si = x_si.view(bs, max_t, self.d)

        sg = sg.view(bs, max_t, self.d)
        mask = Model.get_mask_g_e(bs, max_t, max_e, e_t, e_len, device=sg.device)
        x_e, a_e = self.att_e(sg, e, e, mask=mask.to(sg.device))

        x = x_si + x_e
        x = self.classifier(x).view(bs, max_t)
        return x, a_si.view(bs, max_t, 10), a_e

    @staticmethod
    def loss_fn(pred, x_len, y):  # pred: Float[bs, max_T], y: Float[bs]
        bs, _ = pred.shape
        losses = F.binary_cross_entropy_with_logits(
            pred, y.view(-1, 1).expand_as(pred), reduction='none')
        return torch.stack([torch.mean(losses[i, :x_len[i]])
                            for i in range(bs)]).mean()

    @staticmethod
    def accuracy(pred, y, dur):  # pred: Float[bs, max_T], y: Long[bs], dur: Float[bs]
        pct = torch.FloatTensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to(dur.device)
        pos = (torch.ger(dur, pct) / 60.0).to(torch.long)  # pos: Long[bs, 9]
        pred = torch.gather(pred, dim=1, index=pos)  # pred: Float[bs, 9]
        pred = (pred >= 0).to(torch.long)  # pred: Long[bs, 9]
        cor = (pred == y.unsqueeze(-1).expand_as(pred)).to(torch.long)  # cor: Long[bs, 9]
        return cor.sum().item(), cor.numel()
