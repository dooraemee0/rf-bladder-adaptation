"""
strategies.py -- 통합본 (validation 기반 checkpoint 선택)

0711_2/strategies.py + 0717/strategies_cfp_derpp_v2.py 를 하나로 통합했다.
모든 전략은 다음 시그니처를 따른다:
    train_<name>(model, data, cfg) -> model  (best-val checkpoint 복원 후 반환)

핵심 변경점 (이전 버전과의 차이)
--------------------------------
* checkpoint 선택은 오직 VALIDATION(dev_va/clin_va)으로만 한다.
  기존 _select_best / _log_eval 이 test set(dev_te/clin_te)을 보던 문제를 제거.
  -> ablation_common.evaluate_and_select_best(mode="sum"/"device") 사용.
* trade-off 계열(rehearsal/joint/ewc/si/lwf/coral/mmd/der++/r_derpp/cfp_derpp)은
  dev_va R2 + clin_va R2 최대(sum)로 선택.
* device 계열(device_only / lora_conv_device_only)은 dev_va R2 최대(device)로 선택.
* 학습 종료 후 restore_best 로 best-val checkpoint를 복원. test 평가는 runner에서만.

포함 전략:
  rehearsal, device_only, joint, ewc, si, lwf, coral, mmd,
  der++, r_derpp(제안), cfp_derpp(제안), lora_conv
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ablation_common import (
    DEV, SEED, set_seed, pad_to_400, split_lists, make_loader, evaluate,
    build_rehearsal_df, set_unfreeze_last, count_trainable, last_conv_of,
    evaluate_and_select_best, restore_best, new_best,
)

EVAL_EVERY = 30  # 몇 epoch마다 validation 평가/선택할지


# ---------------------------------------------------------------------------
# 공통 학습 헬퍼
# ---------------------------------------------------------------------------
def _forward(model, batch):
    xh = batch['horizon'].to(DEV); xs = batch['sagittal'].to(DEV)
    x7 = pad_to_400(torch.cat([xh, xs], dim=1))
    rf_h, rf_s = split_lists(x7)
    pred, _, _ = model(rf_h, rf_s)
    y = batch['volume_gt'].to(DEV)
    return pred, y


def _feature_forward(model, batch):
    """concat feature (head 직전) + pred. CORAL/MMD/LwF용."""
    xh = batch['horizon'].to(DEV); xs = batch['sagittal'].to(DEV)
    x7 = pad_to_400(torch.cat([xh, xs], dim=1))
    rf_h, rf_s = split_lists(x7)
    h_feats = [model.h_encoders[i](rf_h[i]) for i in range(model.num_h_rf)]
    s_feats = [model.s_encoders[i](rf_s[i]) for i in range(model.num_s_rf)]
    feat = torch.cat(h_feats + s_feats, dim=1)
    pred = model.head(feat).squeeze(-1)
    return feat, pred


def _make_opt(model, lr):
    return optim.AdamW([p for p in model.parameters() if p.requires_grad],
                       lr=lr, weight_decay=1e-3)


def _should_eval(ep, epochs):
    return (ep % EVAL_EVERY == 0) or (ep == epochs)


# ===========================================================================
# 1. rehearsal — device+clinical 혼합 fine-tuning
# ===========================================================================
def train_rehearsal(model, data, cfg):
    set_unfreeze_last(model, reinit_head=True)
    train_df = build_rehearsal_df(data['dev_tr'], data['clin_tr'], cfg.clinical_ratio)
    print(f"  rehearsal 학습셋: device {len(data['dev_tr'])} + clinical "
          f"{len(train_df)-len(data['dev_tr'])} = {len(train_df)}")
    tl = make_loader(train_df, shuffle=True, pad=True)
    crit = nn.HuberLoss(delta=10.0)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        for batch in tl:
            pred, y = _forward(model, batch)
            opt.zero_grad(); loss = crit(pred, y); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "rehearsal", mode="sum")
    return restore_best(model, best, "rehearsal")


# ===========================================================================
# 2. device_only — device로만 fine-tuning (forgetting 대조군)
# ===========================================================================
def train_device_only(model, data, cfg):
    set_unfreeze_last(model, reinit_head=True)
    tl = make_loader(data['dev_tr'], shuffle=True, pad=True)
    crit = nn.HuberLoss(delta=10.0)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        for batch in tl:
            pred, y = _forward(model, batch)
            opt.zero_grad(); loss = crit(pred, y); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            # device_only는 device validation R2 기준 선택
            evaluate_and_select_best(model, data, best, ep, "device_only", mode="device")
    return restore_best(model, best, "device_only")


# ===========================================================================
# 3. joint — clinical+device 함께 (upper-bound 참조)
# ===========================================================================
def train_joint(model, data, cfg):
    set_unfreeze_last(model, reinit_head=True)
    train_df = build_rehearsal_df(data['dev_tr'], data['clin_tr'],
                                  max(1.0, cfg.clinical_ratio))
    tl = make_loader(train_df, shuffle=True, pad=True)
    crit = nn.HuberLoss(delta=10.0)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        for batch in tl:
            pred, y = _forward(model, batch)
            opt.zero_grad(); loss = crit(pred, y); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "joint", mode="sum")
    return restore_best(model, best, "joint")


# ===========================================================================
# 4. EWC — clinical Fisher penalty
# ===========================================================================
def _compute_fisher(model, loader, crit, n_batches=None):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
              if p.requires_grad}
    model.eval()
    cnt = 0
    for i, batch in enumerate(loader):
        model.zero_grad()
        pred, y = _forward(model, batch)
        loss = crit(pred, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        cnt += 1
        if n_batches and i+1 >= n_batches: break
    for n in fisher: fisher[n] /= max(1, cnt)
    return fisher


def train_ewc(model, data, cfg):
    set_unfreeze_last(model, reinit_head=True)
    crit = nn.HuberLoss(delta=10.0)
    clin_loader = make_loader(data['clin_tr'], shuffle=True, pad=True)
    fisher = _compute_fisher(model, clin_loader, crit)
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    lam = getattr(cfg, 'ewc_lambda', 1e3)

    tl = make_loader(data['dev_tr'], shuffle=True, pad=True)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        for batch in tl:
            pred, y = _forward(model, batch)
            loss = crit(pred, y)
            pen = 0.0
            for n, p in model.named_parameters():
                if p.requires_grad and n in fisher:
                    pen = pen + (fisher[n] * (p - anchor[n]) ** 2).sum()
            loss = loss + lam * pen
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "ewc", mode="sum")
    return restore_best(model, best, "ewc")


# ===========================================================================
# 5. SI (Synaptic Intelligence)
# ===========================================================================
def train_si(model, data, cfg):
    set_unfreeze_last(model, reinit_head=True)
    crit = nn.HuberLoss(delta=10.0)
    lam = getattr(cfg, 'si_lambda', 1.0)
    eps = 1e-3

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    anchor = {n: p.detach().clone() for n, p in params.items()}
    prev = {n: p.detach().clone() for n, p in params.items()}
    w = {n: torch.zeros_like(p) for n, p in params.items()}
    omega = {n: torch.zeros_like(p) for n, p in params.items()}

    tl = make_loader(data['dev_tr'], shuffle=True, pad=True)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        for batch in tl:
            pred, y = _forward(model, batch)
            loss = crit(pred, y)
            pen = 0.0
            for n, p in params.items():
                pen = pen + (omega[n] * (p - anchor[n]) ** 2).sum()
            total = loss + lam * pen
            opt.zero_grad(); total.backward()
            grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
                     for n, p in params.items()}
            opt.step()
            for n, p in params.items():
                if grads[n] is not None:
                    w[n] += -grads[n] * (p.detach() - prev[n])
                prev[n] = p.detach().clone()
        for n, p in params.items():
            omega[n] += torch.clamp(w[n] / ((p.detach()-anchor[n])**2 + eps), min=0)
            w[n].zero_()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "si", mode="sum")
    return restore_best(model, best, "si")


# ===========================================================================
# 6. LwF — clinical 출력 distillation
# ===========================================================================
def train_lwf(model, data, cfg):
    old_model = copy.deepcopy(model).to(DEV).eval()
    for p in old_model.parameters(): p.requires_grad = False
    set_unfreeze_last(model, reinit_head=True)
    crit = nn.HuberLoss(delta=10.0)
    distill = nn.SmoothL1Loss(beta=10.0)
    alpha = getattr(cfg, 'lwf_alpha', 1.0)

    dev_loader = make_loader(data['dev_tr'], shuffle=True, pad=True)
    clin_loader = make_loader(data['clin_tr'], shuffle=True, pad=True)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        clin_iter = iter(clin_loader)
        for batch in dev_loader:
            pred, y = _forward(model, batch)
            loss = crit(pred, y)
            try: cbatch = next(clin_iter)
            except StopIteration:
                clin_iter = iter(clin_loader); cbatch = next(clin_iter)
            with torch.no_grad():
                old_pred, _ = _feature_forward(old_model, cbatch)
            new_pred, _ = _feature_forward(model, cbatch)
            loss = loss + alpha * distill(new_pred, old_pred)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "lwf", mode="sum")
    return restore_best(model, best, "lwf")


# ===========================================================================
# 7. CORAL / MMD — feature-alignment
# ===========================================================================
def _coral_loss(fs, ft):
    d = fs.size(1)
    fs = fs - fs.mean(0, keepdim=True); ft = ft - ft.mean(0, keepdim=True)
    cs = (fs.t() @ fs) / (fs.size(0) - 1 + 1e-6)
    ct = (ft.t() @ ft) / (ft.size(0) - 1 + 1e-6)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def _mmd_loss(fs, ft, sigmas=(1, 2, 4, 8, 16)):
    def rbf(a, b):
        aa = (a**2).sum(1, keepdim=True); bb = (b**2).sum(1, keepdim=True)
        dist = aa + bb.t() - 2 * a @ b.t()
        k = 0.0
        for s in sigmas:
            k = k + torch.exp(-dist / (2 * s**2))
        return k
    return rbf(fs, fs).mean() + rbf(ft, ft).mean() - 2 * rbf(fs, ft).mean()


def _train_feature_align(model, data, cfg, align="coral"):
    set_unfreeze_last(model, reinit_head=True)
    crit = nn.HuberLoss(delta=10.0)
    beta = getattr(cfg, 'align_beta', 1.0)
    dev_loader = make_loader(data['dev_tr'], shuffle=True, pad=True)
    clin_loader = make_loader(data['clin_tr'], shuffle=True, pad=True)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    align_fn = _coral_loss if align == "coral" else _mmd_loss
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        clin_iter = iter(clin_loader)
        for batch in dev_loader:
            ft, pred = _feature_forward(model, batch)
            y = batch['volume_gt'].to(DEV)
            loss = crit(pred, y)
            try: cbatch = next(clin_iter)
            except StopIteration:
                clin_iter = iter(clin_loader); cbatch = next(clin_iter)
            fs, cpred = _feature_forward(model, cbatch)
            cy = cbatch['volume_gt'].to(DEV)
            loss = loss + crit(cpred, cy) + beta * align_fn(fs, ft)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, align, mode="sum")
    return restore_best(model, best, align)


def train_coral(model, data, cfg):
    return _train_feature_align(model, data, cfg, align="coral")


def train_mmd(model, data, cfg):
    return _train_feature_align(model, data, cfg, align="mmd")


# ===========================================================================
# 8b. DER++ (Buzzega et al. 2020, 회귀판)
# ===========================================================================
@torch.no_grad()
def _precompute_old_preds(old_model, df):
    old_model.eval()
    loader = make_loader(df, shuffle=False, pad=True)
    preds = []
    for batch in loader:
        pred, _ = _forward(old_model, batch)
        preds.append(pred.detach().cpu())
    return torch.cat(preds) if preds else torch.empty(0)


def train_der_plus_plus(model, data, cfg):
    old_model = copy.deepcopy(model).to(DEV).eval()
    for p in old_model.parameters(): p.requires_grad = False

    set_unfreeze_last(model, reinit_head=True)
    alpha = getattr(cfg, 'der_alpha', 0.5)
    beta = getattr(cfg, 'der_beta', 0.5)
    crit = nn.HuberLoss(delta=10.0)
    mse = nn.MSELoss()

    dev_loader = make_loader(data['dev_tr'], shuffle=True, pad=True)
    clin_df = data['clin_tr'].reset_index(drop=True)
    z_old_all = _precompute_old_preds(old_model, clin_df)

    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()

    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        clin_loader = make_loader(clin_df, shuffle=False, pad=True)  # 순서 고정(z_old 대응)
        clin_iter = iter(clin_loader)
        clin_start = 0
        for batch in dev_loader:
            pred, y = _forward(model, batch)
            loss = crit(pred, y)
            try:
                cbatch = next(clin_iter); bs = len(cbatch['volume_gt'])
            except StopIteration:
                clin_iter = iter(clin_loader); clin_start = 0
                cbatch = next(clin_iter); bs = len(cbatch['volume_gt'])
            idx = list(range(clin_start, clin_start + bs)); clin_start += bs
            z_old = z_old_all[idx].to(DEV) if len(z_old_all) else None

            cpred, cy = _forward(model, cbatch)
            if z_old is not None and len(z_old) == len(cpred):
                loss = loss + alpha * mse(cpred, z_old)
            loss = loss + beta * crit(cpred, cy)

            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "der++", mode="sum")
    return restore_best(model, best, "der++")


# ===========================================================================
# 8c. R-DER++ (Relation-Preserving DER++) — 제안
# ===========================================================================
def make_valid_pairs(y, min_gap=0.5):
    b = y.numel()
    if b < 2:
        return None, None
    idx_i, idx_j = torch.triu_indices(row=b, col=b, offset=1, device=y.device)
    gap = torch.abs(y[idx_j] - y[idx_i])
    valid = gap >= min_gap
    idx_i, idx_j = idx_i[valid], idx_j[valid]
    if idx_i.numel() == 0:
        return None, None
    return idx_i, idx_j


def pairwise_rank_loss(pred, target, min_gap=0.5, margin=0.1, max_pairs=128):
    idx_i, idx_j = make_valid_pairs(target, min_gap)
    if idx_i is None:
        return pred.new_tensor(0.0), 0
    if idx_i.numel() > max_pairs:
        perm = torch.randperm(idx_i.numel(), device=idx_i.device)[:max_pairs]
        idx_i, idx_j = idx_i[perm], idx_j[perm]
    target_diff = target[idx_j] - target[idx_i]
    pred_diff = pred[idx_j] - pred[idx_i]
    sign = torch.sign(target_diff)
    loss = torch.relu(margin - sign * pred_diff).mean()
    return loss, idx_i.numel()


def pairwise_relation_loss(pred_new, pred_old, target, min_gap=0.5, max_pairs=128):
    idx_i, idx_j = make_valid_pairs(target, min_gap)
    if idx_i is None:
        return pred_new.new_tensor(0.0), 0
    if idx_i.numel() > max_pairs:
        perm = torch.randperm(idx_i.numel(), device=idx_i.device)[:max_pairs]
        idx_i, idx_j = idx_i[perm], idx_j[perm]
    new_diff = pred_new[idx_j] - pred_new[idx_i]
    old_diff = pred_old[idx_j] - pred_old[idx_i]
    loss = torch.nn.functional.smooth_l1_loss(new_diff, old_diff)
    return loss, idx_i.numel()


def train_relation_der_plus_plus(model, data, cfg):
    old_model = copy.deepcopy(model).to(DEV).eval()
    for p in old_model.parameters():
        p.requires_grad = False
    set_unfreeze_last(model, reinit_head=True)

    alpha = getattr(cfg, "der_alpha", 0.5)
    beta = getattr(cfg, "der_beta", 0.5)
    gamma = getattr(cfg, "rank_lambda", 0.05)
    delta = getattr(cfg, "relation_lambda", 0.05)
    pair_min_gap = getattr(cfg, "pair_min_gap", 50.0)
    rank_margin = getattr(cfg, "rank_margin", 10.0)
    scale = 100.0
    min_gap_n = pair_min_gap / scale
    margin_n = rank_margin / scale

    crit = nn.HuberLoss(delta=10.0)
    mse = nn.MSELoss()

    dev_loader = make_loader(data["dev_tr"], shuffle=True, pad=True)
    clin_df = data["clin_tr"].reset_index(drop=True)
    clin_loader = make_loader(clin_df, shuffle=True, pad=True)

    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()

    for ep in range(1, cfg.epochs + 1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        clin_iter = iter(clin_loader)
        agg = {k: [] for k in ["device", "clinical", "dark", "rank", "relation", "total"]}
        n_pairs_log = []

        for dbatch in dev_loader:
            dpred, dy = _forward(model, dbatch)
            loss_device = crit(dpred, dy)

            try:
                cbatch = next(clin_iter)
            except StopIteration:
                clin_iter = iter(clin_loader); cbatch = next(clin_iter)
            cpred_new, cy = _forward(model, cbatch)
            with torch.no_grad():
                cpred_old, _ = _forward(old_model, cbatch)

            loss_clinical = crit(cpred_new, cy)
            loss_dark = mse(cpred_new, cpred_old)

            cpn = cpred_new / scale
            cpo = cpred_old / scale
            cyn = cy / scale
            loss_rank, n_rank = pairwise_rank_loss(cpn, cyn, min_gap_n, margin_n)
            loss_relation, n_rel = pairwise_relation_loss(cpn, cpo, cyn, min_gap_n)

            loss = (loss_device + alpha * loss_dark + beta * loss_clinical
                    + gamma * loss_rank + delta * loss_relation)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0)
            opt.step()

            agg["device"].append(loss_device.item())
            agg["clinical"].append(loss_clinical.item())
            agg["dark"].append(loss_dark.item())
            agg["rank"].append(float(loss_rank))
            agg["relation"].append(float(loss_relation))
            agg["total"].append(loss.item())
            n_pairs_log.append(n_rank)
        sched.step()

        if _should_eval(ep, cfg.epochs):
            mp = float(np.mean(n_pairs_log)) if n_pairs_log else 0.0
            print(f"  [r_derpp E{ep:3d}] total={np.mean(agg['total']):.4f} "
                  f"dev={np.mean(agg['device']):.3f} clin={np.mean(agg['clinical']):.3f} "
                  f"dark={np.mean(agg['dark']):.3f} rank={np.mean(agg['rank']):.4f} "
                  f"rel={np.mean(agg['relation']):.4f} | pairs/batch={mp:.0f}")
            evaluate_and_select_best(model, data, best, ep, "r_derpp", mode="sum")
    return restore_best(model, best, "r_derpp")


# ===========================================================================
# 8d. CFP-DER++ (Channel-wise Feature-Preserving DER++) — 제안
# ===========================================================================
def _channel_feature_forward(model, batch):
    """각 채널 encoder feature(list of BxD)와 최종 예측. 순서 [H1..H3, S1..S4]."""
    xh = batch['horizon'].to(DEV)
    xs = batch['sagittal'].to(DEV)
    x7 = pad_to_400(torch.cat([xh, xs], dim=1))
    rf_h, rf_s = split_lists(x7)
    h_feats = [model.h_encoders[i](rf_h[i]) for i in range(model.num_h_rf)]
    s_feats = [model.s_encoders[i](rf_s[i]) for i in range(model.num_s_rf)]
    channel_feats = h_feats + s_feats
    fused = torch.cat(channel_feats, dim=1)
    pred = model.head(fused).squeeze(-1)
    return channel_feats, pred


def _channel_feature_loss(new_feats, old_feats, mode='cosine', eps=1e-8,
                          channel_weights=None):
    if len(new_feats) != len(old_feats):
        raise ValueError(f'channel count mismatch: new={len(new_feats)} old={len(old_feats)}')
    n_ch = len(new_feats)
    if channel_weights is None:
        weights = [1.0 / max(1, n_ch)] * n_ch
    else:
        w = torch.as_tensor(channel_weights, dtype=new_feats[0].dtype,
                            device=new_feats[0].device)
        if w.numel() != n_ch:
            raise ValueError(f'feature_channel_weights must have {n_ch} values, got {w.numel()}')
        w = w / (w.sum() + eps)
        weights = [w[i] for i in range(n_ch)]

    total = new_feats[0].new_tensor(0.0)
    per_channel = []
    for k, (f_new, f_old) in enumerate(zip(new_feats, old_feats)):
        f_new = f_new.flatten(start_dim=1)
        f_old = f_old.detach().flatten(start_dim=1)
        new_n = torch.nn.functional.normalize(f_new, p=2, dim=1, eps=eps)
        old_n = torch.nn.functional.normalize(f_old, p=2, dim=1, eps=eps)
        cos = (1.0 - (new_n * old_n).sum(dim=1)).mean()
        new_z = (f_new - f_new.mean(dim=1, keepdim=True)) / (
            f_new.std(dim=1, keepdim=True, unbiased=False) + eps)
        old_z = (f_old - f_old.mean(dim=1, keepdim=True)) / (
            f_old.std(dim=1, keepdim=True, unbiased=False) + eps)
        sl1 = torch.nn.functional.smooth_l1_loss(new_z, old_z)
        if mode == 'cosine':
            lk = cos
        elif mode == 'smooth_l1':
            lk = sl1
        elif mode == 'hybrid':
            lk = cos + 0.1 * sl1
        else:
            raise ValueError(f'unknown feature_loss_mode: {mode}')
        total = total + weights[k] * lk
        per_channel.append(float(lk.detach().cpu()))
    return total, per_channel


def _feature_lambda_at_epoch(base_lambda, ep, epochs, schedule='constant', min_ratio=0.1):
    if base_lambda <= 0:
        return 0.0
    progress = 0.0 if epochs <= 1 else (ep - 1) / (epochs - 1)
    min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
    if schedule == 'constant':
        ratio = 1.0
    elif schedule == 'linear_decay':
        ratio = 1.0 - (1.0 - min_ratio) * progress
    elif schedule == 'cosine_decay':
        ratio = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
    elif schedule == 'linear_warmup':
        ratio = min_ratio + (1.0 - min_ratio) * progress
    elif schedule == 'cosine_warmup':
        ratio = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 - np.cos(np.pi * progress))
    elif schedule == 'early_half':
        ratio = 1.0 if progress < 0.5 else 0.0
    elif schedule == 'late_half':
        ratio = 0.0 if progress < 0.5 else 1.0
    else:
        raise ValueError(f'unknown feature_schedule: {schedule}')
    return float(base_lambda * ratio)


def train_channel_feature_der_plus_plus(model, data, cfg):
    old_model = copy.deepcopy(model).to(DEV).eval()
    for p in old_model.parameters():
        p.requires_grad = False
    set_unfreeze_last(model, reinit_head=True)

    alpha = getattr(cfg, 'der_alpha', 0.5)
    beta = getattr(cfg, 'der_beta', 0.5)
    feature_lambda = getattr(cfg, 'feature_lambda', 0.05)
    feature_mode = getattr(cfg, 'feature_loss_mode', 'cosine')
    feature_schedule = getattr(cfg, 'feature_schedule', 'constant')
    feature_min_ratio = getattr(cfg, 'feature_min_ratio', 0.1)
    grad_clip = getattr(cfg, 'feature_grad_clip', 5.0)

    raw_weights = getattr(cfg, 'feature_channel_weights', '')
    channel_weights = None
    if raw_weights:
        channel_weights = [float(v.strip()) for v in raw_weights.split(',') if v.strip()]

    crit = nn.HuberLoss(delta=10.0)
    mse = nn.MSELoss()

    dev_loader = make_loader(data['dev_tr'], shuffle=True, pad=True)
    clin_df = data['clin_tr'].reset_index(drop=True)
    clin_loader = make_loader(clin_df, shuffle=True, pad=True)

    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    channel_names = [f'H{i+1}' for i in range(model.num_h_rf)] + [
        f'S{i+1}' for i in range(model.num_s_rf)]

    # 채널별 feature distance를 best-val checkpoint 시점에 저장 (논문 분석용)
    last_ch_feat = {name: float('nan') for name in channel_names}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        model.h_encoders.eval(); model.s_encoders.eval()
        old_model.eval()

        clin_iter = iter(clin_loader)
        agg = {k: [] for k in ['device', 'clinical', 'dark', 'feature', 'total']}
        ch_agg = [[] for _ in channel_names]
        lam_ep = _feature_lambda_at_epoch(
            feature_lambda, ep, cfg.epochs, feature_schedule, feature_min_ratio)

        for dbatch in dev_loader:
            dpred, dy = _forward(model, dbatch)
            loss_device = crit(dpred, dy)

            try:
                cbatch = next(clin_iter)
            except StopIteration:
                clin_iter = iter(clin_loader)
                cbatch = next(clin_iter)

            new_feats, cpred_new = _channel_feature_forward(model, cbatch)
            cy = cbatch['volume_gt'].to(DEV)
            with torch.no_grad():
                old_feats, cpred_old = _channel_feature_forward(old_model, cbatch)

            loss_clinical = crit(cpred_new, cy)
            loss_dark = mse(cpred_new, cpred_old)
            loss_feature, per_ch = _channel_feature_loss(
                new_feats, old_feats, mode=feature_mode,
                channel_weights=channel_weights)

            loss = (loss_device
                    + alpha * loss_dark
                    + beta * loss_clinical
                    + lam_ep * loss_feature)

            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip)
            opt.step()

            agg['device'].append(loss_device.item())
            agg['clinical'].append(loss_clinical.item())
            agg['dark'].append(loss_dark.item())
            agg['feature'].append(loss_feature.item())
            agg['total'].append(loss.item())
            for k, val in enumerate(per_ch):
                ch_agg[k].append(val)
        sched.step()

        if _should_eval(ep, cfg.epochs):
            ch_msg = ' '.join(
                f'{name}={np.mean(vals):.4f}' if vals else f'{name}=nan'
                for name, vals in zip(channel_names, ch_agg))
            print(f"  [cfp_derpp E{ep:3d}] total={np.mean(agg['total']):.4f} "
                  f"dev={np.mean(agg['device']):.3f} clin={np.mean(agg['clinical']):.3f} "
                  f"dark={np.mean(agg['dark']):.3f} feat={np.mean(agg['feature']):.4f} "
                  f"lambda_f={lam_ep:.5f} mode={feature_mode} | {ch_msg}")
            _, sel = evaluate_and_select_best(model, data, best, ep, "cfp_derpp", mode="sum")
            # best가 이 epoch에서 갱신됐다면 채널 feature도 기록
            if best.get('epoch') == ep:
                for name, vals in zip(channel_names, ch_agg):
                    last_ch_feat[name] = float(np.mean(vals)) if vals else float('nan')

    model = restore_best(model, best, "cfp_derpp")
    # 채널 feature distance를 모델 속성으로 부착 (runner가 metrics에 기록)
    model._cfp_channel_feature = last_ch_feat
    return model


# ===========================================================================
# LoRA-conv
# ===========================================================================
class Conv1dLoRA(nn.Module):
    """기존 Conv1d를 감싸 low-rank 보정을 더하는 wrapper.
    out = base(x) + scale * B(A(x)). base freeze, A/B만 학습."""
    def __init__(self, base_conv, rank=4, alpha=8.0):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters(): p.requires_grad = False
        cin = base_conv.in_channels; cout = base_conv.out_channels
        k = base_conv.kernel_size[0]; pad = base_conv.padding[0]
        stride = base_conv.stride[0]
        self.A = nn.Conv1d(cin, rank, kernel_size=k, padding=pad, stride=stride, bias=False)
        self.B = nn.Conv1d(rank, cout, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.A.weight); nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def _inject_lora(model, rank=4, alpha=8.0):
    for enc in list(model.h_encoders) + list(model.s_encoders):
        idxs = [i for i, m in enumerate(enc.conv_block) if isinstance(m, nn.Conv1d)]
        if not idxs: continue
        li = idxs[-1]
        enc.conv_block[li] = Conv1dLoRA(enc.conv_block[li], rank=rank, alpha=alpha).to(DEV)
    return model


def train_lora_conv(model, data, cfg):
    for p in model.parameters(): p.requires_grad = False
    rank = getattr(cfg, 'lora_rank', 4); alpha = getattr(cfg, 'lora_alpha', 8.0)
    _inject_lora(model, rank=rank, alpha=alpha)
    for m in model.head:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
    for p in model.head.parameters(): p.requires_grad = True

    train_df = build_rehearsal_df(data['dev_tr'], data['clin_tr'], cfg.clinical_ratio)
    tl = make_loader(train_df, shuffle=True, pad=True)
    crit = nn.HuberLoss(delta=10.0)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        for batch in tl:
            pred, y = _forward(model, batch)
            opt.zero_grad(); loss = crit(pred, y); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "lora_conv", mode="sum")
    return restore_best(model, best, "lora_conv")


def train_lora_conv_device_only(model, data, cfg):
    for p in model.parameters(): p.requires_grad = False
    rank = getattr(cfg, 'lora_rank', 4); alpha = getattr(cfg, 'lora_alpha', 8.0)
    _inject_lora(model, rank=rank, alpha=alpha)
    for m in model.head:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
    for p in model.head.parameters(): p.requires_grad = True
    tl = make_loader(data['dev_tr'], shuffle=True, pad=True)
    crit = nn.HuberLoss(delta=10.0)
    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()
    for ep in range(1, cfg.epochs+1):
        model.train(); model.h_encoders.eval(); model.s_encoders.eval()
        for batch in tl:
            pred, y = _forward(model, batch)
            opt.zero_grad(); loss = crit(pred, y); loss.backward(); opt.step()
        sched.step()
        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, "lora_conv[dev-only]", mode="device")
    return restore_best(model, best, "lora_conv[dev-only]")


# 각 방법의 "clinic 없는(device-only)" 버전 매핑
NO_CLINIC_VARIANT = {
    "rehearsal":   train_device_only,
    "joint":       train_device_only,
    "ewc":         train_device_only,
    "si":          train_device_only,
    "lwf":         train_device_only,
    "coral":       train_device_only,
    "mmd":         train_device_only,
    "der++":       train_device_only,
    "r_derpp":     train_device_only,
    "cfp_derpp":   train_device_only,
    "lora_conv":   train_lora_conv_device_only,
    "device_only": train_device_only,
}


# ---------------------------------------------------------------------------
# 전략 레지스트리
# ---------------------------------------------------------------------------
STRATEGIES = {
    "rehearsal":   train_rehearsal,
    "device_only": train_device_only,
    "joint":       train_joint,
    "ewc":         train_ewc,
    "si":          train_si,
    "lwf":         train_lwf,
    "coral":       train_coral,
    "mmd":         train_mmd,
    "der++":       train_der_plus_plus,
    "r_derpp":     train_relation_der_plus_plus,
    "cfp_derpp":   train_channel_feature_der_plus_plus,
    "lora_conv":   train_lora_conv,
}
