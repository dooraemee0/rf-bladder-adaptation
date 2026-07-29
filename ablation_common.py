"""
ablation_common.py -- 통합본 (validation-split + RNG-safe evaluate)

이 파일은 0711_2 / 0717 두 버전을 통합하고, 다음 세 가지 방법론적 결함을 고친다.

  (1) [핵심] checkpoint 선택이 test set(dev_te/clin_te)을 직접 보고 이뤄지던 문제.
      -> validation split(dev_va/clin_va)을 새로 도입하고, checkpoint 선택은
         오직 validation으로만 한다. test는 최종 1회만 평가한다.

  (2) evaluate()가 내부에서 set_seed()를 호출해 학습 도중 전역 RNG를 리셋하던 문제.
      -> evaluate()는 진입 시 RNG 상태를 저장하고, 종료 시 복원한다. 따라서
         "학습 중간 평가"가 이후 학습의 난수열을 오염시키지 않는다. (seed 재현성)

  (3) clinical validation 분리 시 같은 환자의 acquisition이 train/val에 동시에
      들어가면 leakage가 된다.
      -> clinical split은 환자(파일 prefix) 단위(group split)로 수행한다.
         (명시적 patient id 컬럼이 없으면 Upright_H prefix로 patient key를 유도)

경로/상수는 환경변수로 override 가능하다. 기존 스크립트와 호환된다.
"""
import os, glob, random, re, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import RFNet
from dataloader import FolderDatasetUnified, PreprocessTransform

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------
SEED = 42
DEVICE_BASE = os.environ.get("ABL_DEVICE_BASE", "")
SNUH_EXCEL = os.environ.get("ABL_SNUH_EXCEL", "")
SNUH_DATA = os.environ.get("ABL_SNUH_DATA", "")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LEN = 400

# validation 분리 비율 (device는 train 대비, clinical은 train(=non-test) 대비)
DEV_VAL_RATIO = float(os.environ.get("ABL_DEV_VAL_RATIO", "0.2"))
CLIN_VAL_RATIO = float(os.environ.get("ABL_CLIN_VAL_RATIO", "0.2"))


# ----------------------------------------------------------------------------
# 시드 / 텐서 헬퍼
# ----------------------------------------------------------------------------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_snapshot():
    """python/numpy/torch(cpu,cuda) RNG 상태를 한 번에 저장."""
    return dict(
        py=random.getstate(),
        np=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    )


def _rng_restore(state):
    random.setstate(state['py'])
    np.random.set_state(state['np'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def pad_to_400(x):
    """(B,C,L) 배치 텐서를 (B,C,400)으로."""
    B, C, L = x.shape
    if L == MODEL_LEN: return x
    if L > MODEL_LEN: return x[:, :, :MODEL_LEN]
    tail = min(30, L)
    tm = x[:, :, -tail:].mean(dim=2, keepdim=True)
    return torch.cat([x, tm.expand(B, C, MODEL_LEN - L)], dim=2)


def split_lists(x7):
    """(B,7,L) -> (h list[3], s list[4])."""
    return [x7[:, i:i+1, :] for i in range(3)], [x7[:, i:i+1, :] for i in range(3, 7)]


def _pad_seq_to_400(seq):
    C, L = seq.shape
    if L == MODEL_LEN: return seq
    if L > MODEL_LEN: return seq[:, :MODEL_LEN]
    tail = min(30, L)
    tm = seq[:, -tail:].mean(dim=1, keepdim=True)
    return torch.cat([seq, tm.expand(C, MODEL_LEN - L)], dim=1)


def collate_pad400(batch):
    for d in batch:
        d['horizon'] = _pad_seq_to_400(d['horizon'])
        d['sagittal'] = _pad_seq_to_400(d['sagittal'])
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


# ----------------------------------------------------------------------------
# 데이터프레임 빌드
# ----------------------------------------------------------------------------
def build_device_df():
    rows = []
    for folder in ["50", "150", "300"]:
        for txt in sorted(glob.glob(os.path.join(DEVICE_BASE, folder, "*.txt"))):
            rows.append({"data_path": DEVICE_BASE, "txt_path": txt, "domain": "device",
                         "volume": float(folder), "Upright_H": "", "Upright_S": ""})
    return pd.DataFrame(rows)


def _patient_key(upright_h):
    """clinical 파일 prefix에서 환자 키를 유도.

    같은 환자의 여러 acquisition이 공통 prefix를 공유한다는 가정 하에,
    파일명 앞부분의 '숫자/식별자'를 patient key로 사용한다.
    예) 'P012_H_upright_1' -> 'P012'  /  '12-3' -> '12'
    데이터 명명 규칙이 다르면 아래 정규식만 바꾸면 된다.
    """
    s = str(upright_h).strip()
    if not s:
        return ""
    # 1) 흔한 케이스: 앞의 영문+숫자 토큰 (P012, S3, ID45 등)
    m = re.match(r'^([A-Za-z]*\d+)', s)
    if m:
        return m.group(1)
    # 2) 구분자(_,-,공백) 앞부분
    return re.split(r'[_\-\s]', s)[0]


def build_snuh_df(test_idx=None, want_test=False):
    df = pd.read_csv(SNUH_EXCEL); df.rename(columns=lambda c: c.strip(), inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce'); df.dropna(subset=['volume'], inplace=True)
    df['original_index'] = df.index
    rows = []
    for _, row in df.iterrows():
        h = glob.glob(os.path.join(SNUH_DATA, f"{row['Upright_H']}*.csv"))
        s = glob.glob(os.path.join(SNUH_DATA, f"{row['Upright_S']}*.csv"))
        if not h or not s: continue
        rows.append({"data_path": SNUH_DATA, "txt_path": "", "domain": "SNUH_uncut",
                     "volume": float(row['volume']), "Upright_H": row['Upright_H'],
                     "Upright_S": row['Upright_S'], "original_index": row['original_index']})
    sdf = pd.DataFrame(rows); sdf = sdf[sdf['volume'] <= 700].reset_index(drop=True)
    # patient key 부여 (group split용)
    sdf['patient'] = sdf['Upright_H'].map(_patient_key)
    if test_idx is not None:
        te = sdf[sdf['original_index'].isin(test_idx)].reset_index(drop=True)
        tr = sdf[~sdf['original_index'].isin(test_idx)].reset_index(drop=True)
        return (te if want_test else tr)
    return sdf


# ----------------------------------------------------------------------------
# split: device(volume-stratified) / clinical(patient-group)
# ----------------------------------------------------------------------------
def split_device(df, test_ratio=0.2):
    """volume별로 test_ratio를 떼어 (train, test) 반환. (원본과 동일 동작)"""
    tr, te = [], []
    for vol in sorted(df['volume'].unique()):
        v = df[df['volume'] == vol]
        a, b = train_test_split(v.index, test_size=test_ratio, random_state=SEED, shuffle=True)
        tr.append(df.loc[a]); te.append(df.loc[b])
    return pd.concat(tr).reset_index(drop=True), pd.concat(te).reset_index(drop=True)


def split_device_trainval(dev_tr, val_ratio=DEV_VAL_RATIO, seed=SEED):
    """device train을 다시 (train, val)로 volume-stratified 분리.
    각 volume(50/150/300)이 val에도 동일 비율로 포함되도록 한다."""
    tr, va = [], []
    for vol in sorted(dev_tr['volume'].unique()):
        v = dev_tr[dev_tr['volume'] == vol]
        if len(v) < 2:
            tr.append(v); continue
        a, b = train_test_split(v.index, test_size=val_ratio, random_state=seed, shuffle=True)
        tr.append(dev_tr.loc[a]); va.append(dev_tr.loc[b])
    tr_df = pd.concat(tr).reset_index(drop=True)
    va_df = pd.concat(va).reset_index(drop=True) if va else dev_tr.iloc[0:0].copy()
    return tr_df, va_df


def split_clinical_trainval(clin_tr, val_ratio=CLIN_VAL_RATIO, seed=SEED):
    """clinical train을 (train, val)로 환자(patient) 그룹 단위 분리.
    같은 환자의 acquisition이 train/val에 동시에 들어가지 않도록 보장한다."""
    df = clin_tr.reset_index(drop=True).copy()
    if 'patient' not in df.columns or df['patient'].eq("").all():
        # patient key가 없으면 (안전 fallback) acquisition 단위 분리 + 경고
        print("[WARN] clinical patient key 없음 -> acquisition 단위 val 분리로 대체")
        a, b = train_test_split(df.index, test_size=val_ratio, random_state=seed, shuffle=True)
        return df.loc[a].reset_index(drop=True), df.loc[b].reset_index(drop=True)

    patients = df['patient'].unique().tolist()
    if len(patients) < 2:
        print("[WARN] clinical 환자 수 < 2 -> val 분리 불가, 전체를 train으로 사용")
        return df, df.iloc[0:0].copy()

    # 환자 단위로 val_ratio 만큼 환자를 통째로 val에 배정
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(patients))
    n_val = max(1, int(round(len(patients) * val_ratio)))
    val_patients = set(np.array(patients)[perm[:n_val]].tolist())
    va = df[df['patient'].isin(val_patients)].reset_index(drop=True)
    tr = df[~df['patient'].isin(val_patients)].reset_index(drop=True)
    return tr, va


# ----------------------------------------------------------------------------
# 데이터 로더
# ----------------------------------------------------------------------------
_NUM_WORKERS = int(os.environ.get("ABL_NUM_WORKERS", "0"))
_EVAL_NUM_WORKERS = int(os.environ.get("ABL_EVAL_NUM_WORKERS", str(_NUM_WORKERS)))
_PIN_MEMORY = os.environ.get("ABL_PIN_MEMORY", "0") == "1"
_CACHE_DIR = os.environ.get("ABL_CACHE_DIR", "./_prep_cache")
_USE_CACHE = os.environ.get("ABL_USE_CACHE", "0") == "1"


class CachedDataset(torch.utils.data.Dataset):
    """__getitem__ 결과를 디스크(.npz)에 캐싱 (결정적 전처리라 재현성 보존)."""
    def __init__(self, base, tag):
        self.base = base
        self.cache_sub = os.path.join(_CACHE_DIR, tag)
        os.makedirs(self.cache_sub, exist_ok=True)

    def __len__(self):
        return len(self.base)

    def _key_path(self, idx):
        return os.path.join(self.cache_sub, f"item_{idx}.npz")

    def __getitem__(self, idx):
        path = self._key_path(idx)
        if os.path.exists(path):
            try:
                d = np.load(path, allow_pickle=True)
                return {
                    'horizon': torch.from_numpy(d['horizon']),
                    'sagittal': torch.from_numpy(d['sagittal']),
                    'volume_gt': torch.tensor(float(d['volume_gt'])),
                    'domain': str(d['domain']),
                }
            except Exception:
                pass
        item = self.base[idx]
        try:
            np.savez(path,
                     horizon=item['horizon'].numpy(),
                     sagittal=item['sagittal'].numpy(),
                     volume_gt=float(item['volume_gt']),
                     domain=str(item['domain']))
        except Exception:
            pass
        return item


def _dataset_tag(df):
    key = f"{len(df)}_" + "_".join(
        f"{r['domain']}:{r.get('txt_path','')}:{r.get('Upright_H','')}:{r['volume']}"
        for _, r in df.iterrows())
    return hashlib.md5(key.encode()).hexdigest()[:16]


def make_loader(df, shuffle, bs=16, pad=False):
    th = PreprocessTransform(num_select=3); ts = PreprocessTransform(num_select=4)
    ds = FolderDatasetUnified(df.reset_index(drop=True), th, ts)
    if _USE_CACHE:
        ds = CachedDataset(ds, tag=_dataset_tag(df.reset_index(drop=True)))
    cf = collate_pad400 if pad else None
    kw = {}
    if _NUM_WORKERS > 0:
        kw = dict(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=_NUM_WORKERS,
                      collate_fn=cf, pin_memory=_PIN_MEMORY, **kw)


def load_all_data(test_idx):
    """device / clinical 각각 train/val/test 를 한 번에 준비.

    반환 dict keys:
      dev_tr, dev_va, dev_te, clin_tr, clin_va, clin_te
    (val은 checkpoint 선택 전용, test는 최종 평가 전용)
    """
    device_df = build_device_df()
    dev_tr_full, dev_te = split_device(device_df)
    dev_tr, dev_va = split_device_trainval(dev_tr_full)

    clin_tr_full = build_snuh_df(test_idx=test_idx, want_test=False)
    clin_te = build_snuh_df(test_idx=test_idx, want_test=True)
    clin_tr, clin_va = split_clinical_trainval(clin_tr_full)

    return dict(dev_tr=dev_tr, dev_va=dev_va, dev_te=dev_te,
                clin_tr=clin_tr, clin_va=clin_va, clin_te=clin_te)


# ----------------------------------------------------------------------------
# 평가 (RNG-safe: 진입 시 상태 저장, 종료 시 복원)
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, df, n_aug=2):
    """채널 랜덤 n_aug 평균 평가.

    주의: 내부에서 set_seed로 augmentation을 결정적으로 만들지만,
    함수 진입 전 RNG 상태를 저장했다가 종료 시 복원하므로
    학습 루프의 난수열에는 영향을 주지 않는다 (seed 재현성 보존).
    """
    rng_state = _rng_snapshot()
    try:
        model.eval()
        passes = []; gt_ref = None
        eval_nw = _EVAL_NUM_WORKERS
        for so in range(n_aug):
            set_seed(SEED + so*1000)
            th = PreprocessTransform(num_select=3, seed_offset=so*1000)
            ts = PreprocessTransform(num_select=4, seed_offset=so*1000)
            ds = FolderDatasetUnified(df.reset_index(drop=True), th, ts)
            dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=eval_nw,
                            pin_memory=_PIN_MEMORY)
            P, G = [], []
            for batch in dl:
                xh = batch['horizon'].to(DEV); xs = batch['sagittal'].to(DEV)
                x7 = pad_to_400(torch.cat([xh, xs], dim=1))
                rf_h, rf_s = split_lists(x7)
                pred, _, _ = model(rf_h, rf_s)
                P.extend(pred.cpu().numpy().flatten()); G.extend(batch['volume_gt'].numpy().flatten())
            passes.append(np.array(P)); gt_ref = np.array(G) if gt_ref is None else gt_ref
        pred = np.mean(np.stack(passes), 0); gt = gt_ref
        return dict(pred=pred, gt=gt,
                    mae=float(mean_absolute_error(gt, pred)),
                    rmse=float(np.sqrt(mean_squared_error(gt, pred))),
                    r2=float(r2_score(gt, pred)) if len(np.unique(gt)) > 1 else float('nan'),
                    acc50=float(np.mean(np.abs(pred-gt) <= 50)*100))
    finally:
        _rng_restore(rng_state)


# ----------------------------------------------------------------------------
# validation 기반 checkpoint 선택 (모든 전략 공통)
# ----------------------------------------------------------------------------
def evaluate_and_select_best(model, data, best, epoch, tag="", mode="sum"):
    """VALIDATION set으로만 평가하여 best checkpoint를 갱신한다.

    mode:
      - "sum"    : dev_va R2 + clin_va R2 최대 (device-clinical trade-off 방법들)
      - "device" : dev_va R2 최대 (device_only 계열)

    test set은 절대 참조하지 않는다. 학습 종료 후 test는 run_one에서 1회만 평가.
    반환: (rd_val, rc_val) — 로깅/디버깅용 validation 결과.
    """
    rd = evaluate(model, data['dev_va'])
    if mode == "device":
        rc = None
        score = rd['r2']
    else:
        rc = evaluate(model, data['clin_va'])
        score = rd['r2'] + (rc['r2'] if rc['r2'] == rc['r2'] else -1e9)  # nan 방어

    if score > best['score']:
        best['score'] = score
        best['epoch'] = epoch
        best['device_val_r2'] = rd['r2']
        best['clinical_val_r2'] = (rc['r2'] if rc is not None else float('nan'))
        best['state'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if rc is not None:
        print(f"  [{tag} E{epoch:3d}] VAL device R2 {rd['r2']:.3f} (MAE {rd['mae']:.1f}) | "
              f"VAL clinical R2 {rc['r2']:.3f} (MAE {rc['mae']:.1f}) | sum {score:.3f}")
    else:
        print(f"  [{tag} E{epoch:3d}] VAL device R2 {rd['r2']:.3f} (MAE {rd['mae']:.1f})")
    return rd, rc


def restore_best(model, best, tag=""):
    """학습 종료 후 best validation checkpoint를 복원."""
    if best.get('state') is None:
        print(f"  [WARN] {tag}: validation checkpoint가 선택되지 않아 최종 epoch 모델 사용")
        return model
    model.load_state_dict(best['state'])
    print(f"  [best-val checkpoint] {tag} epoch={best.get('epoch')} | "
          f"dev_val_R2={best.get('device_val_r2', float('nan')):.3f} | "
          f"clin_val_R2={best.get('clinical_val_r2', float('nan')):.3f} | "
          f"sum={best['score']:.3f}")
    return model


def new_best():
    return dict(score=-1e9, state=None, epoch=None,
                device_val_r2=float('nan'), clinical_val_r2=float('nan'))


# ----------------------------------------------------------------------------
# bootstrap / per-volume / tail (test 최종 리포트용)
# ----------------------------------------------------------------------------
def bootstrap_metrics(res, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    pred, gt = res['pred'], res['gt']
    n = len(gt)
    maes, rmses, r2s, accs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        p, g = pred[idx], gt[idx]
        maes.append(mean_absolute_error(g, p))
        rmses.append(np.sqrt(mean_squared_error(g, p)))
        r2s.append(r2_score(g, p) if len(np.unique(g)) > 1 else np.nan)
        accs.append(np.mean(np.abs(p-g) <= 50)*100)
    def ms(a):
        a = np.asarray(a, float); a = a[~np.isnan(a)]
        return float(np.mean(a)), float(np.std(a))
    mae_m, mae_s = ms(maes); rmse_m, rmse_s = ms(rmses)
    r2_m, r2_s = ms(r2s); acc_m, acc_s = ms(accs)
    return dict(mae=mae_m, mae_std=mae_s, rmse=rmse_m, rmse_std=rmse_s,
                r2=r2_m, r2_std=r2_s, acc50=acc_m, acc50_std=acc_s)


def per_volume_pred(res):
    out = {}
    for vol in sorted(np.unique(res['gt'])):
        m = res['gt'] == vol
        out[float(vol)] = dict(mean=float(res['pred'][m].mean()),
                               std=float(res['pred'][m].std()),
                               n=int(m.sum()))
    return out


def tail_volume_mae(res, low_q=0.2, high_q=0.8):
    gt = res['gt']; pred = res['pred']
    lo, hi = np.quantile(gt, low_q), np.quantile(gt, high_q)
    mask = (gt < lo) | (gt > hi)
    if mask.sum() == 0:
        return float('nan')
    return float(mean_absolute_error(gt[mask], pred[mask]))


# ----------------------------------------------------------------------------
# 모델 생성 / 로드 / freeze
# ----------------------------------------------------------------------------
def fresh_model():
    return RFNet(num_h_rf=3, num_s_rf=4, encoder_out_dim=512, dropout_p=0.5).to(DEV)


def load_clinical_model(ckpt):
    model = fresh_model()
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=False))
    return model


def last_conv_of(enc):
    convs = [m for m in enc.conv_block if isinstance(m, nn.Conv1d)]
    return convs[-1] if convs else None


def set_unfreeze_last(model, reinit_head=True):
    for p in model.h_encoders.parameters(): p.requires_grad = False
    for p in model.s_encoders.parameters(): p.requires_grad = False
    for enc in list(model.h_encoders) + list(model.s_encoders):
        c = last_conv_of(enc)
        if c is not None:
            for p in c.parameters(): p.requires_grad = True
    if reinit_head:
        for m in model.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)


def set_unfreeze_scope(model, scope, reinit_head=True):
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    if scope == "head_only":
        pass
    elif scope == "last_conv":
        for enc in list(model.h_encoders) + list(model.s_encoders):
            c = last_conv_of(enc)
            if c is not None:
                for p in c.parameters(): p.requires_grad = True
    elif scope == "all_conv":
        for p in model.h_encoders.parameters(): p.requires_grad = True
        for p in model.s_encoders.parameters(): p.requires_grad = True
    else:
        raise ValueError(f"unknown scope {scope}")
    if reinit_head:
        for m in model.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)


def count_trainable(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


# ----------------------------------------------------------------------------
# 리허설용 학습 데이터프레임 구성
# ----------------------------------------------------------------------------
def build_rehearsal_df(dev_tr, clin_tr, clinical_ratio):
    if clinical_ratio <= 0:
        return dev_tr.copy()
    if clinical_ratio >= 1:
        rep = pd.concat([clin_tr] * int(round(clinical_ratio)), ignore_index=True)
    else:
        rep = clin_tr.sample(frac=clinical_ratio, random_state=SEED)
    return pd.concat([dev_tr, rep], ignore_index=True)


def rehearsal_composition(dev_tr, clin_tr, clinical_ratio):
    n_dev = len(dev_tr)
    if clinical_ratio <= 0:
        n_clin = 0
    elif clinical_ratio >= 1:
        n_clin = len(clin_tr) * int(round(clinical_ratio))
    else:
        n_clin = int(round(len(clin_tr) * clinical_ratio))
    total = n_dev + n_clin
    return dict(n_device=n_dev, n_clinical=n_clin, total=total,
                clinical_frac=round(n_clin / total, 4) if total else 0.0,
                device_to_clinical=round(n_dev / n_clin, 4) if n_clin else float('inf'))
