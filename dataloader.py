import os, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d

# ---------------------- 향상된 전처리 클래스 ----------------------
class PreprocessTransformEnhanced:
    """
    Enhanced RF 데이터 전처리:
    - 더 정교한 envelope 처리
    - 적응적 노이즈 감소
    - 다양한 증강 기법
    """
    def __init__(self, num_select=None, sigma=1, downsample_rate=10, max_len=220, 
                 seed_offset=10, use_adaptive_envelope=True, noise_reduction=True):
        self.num_select = num_select
        self.sigma = sigma
        self.downsample_rate = downsample_rate
        self.default_max_len = max_len
        self.seed_offset = seed_offset
        self.use_adaptive_envelope = use_adaptive_envelope
        self.noise_reduction = noise_reduction

    def adaptive_envelope_extraction(self, data):
        """적응적 envelope 추출"""
        envelope_data = np.zeros_like(data)
        
        for ch in range(data.shape[0]):
            signal = data[ch]
            
            # Hilbert envelope
            analytic_signal = hilbert(signal)
            envelope = np.abs(analytic_signal)
            
            if self.use_adaptive_envelope:
                # 적응적 threshold로 노이즈 감소
                envelope_mean = np.mean(envelope)
                envelope_std = np.std(envelope)
                threshold = envelope_mean + 0.5 * envelope_std
                
                # Soft thresholding
                envelope = np.where(envelope > threshold, 
                                  envelope, 
                                  envelope * (envelope / threshold) ** 2)
                
                # 추가 스무딩으로 더 부드러운 envelope
                from scipy.ndimage import gaussian_filter1d
                envelope = gaussian_filter1d(envelope, sigma=1.5)
            
            envelope_data[ch] = envelope
            
        return envelope_data

    def noise_reduction_filter(self, data, domain):
        """도메인별 적응적 노이즈 감소"""
        if not self.noise_reduction:
            return data
            
        filtered_data = np.zeros_like(data)
        
        for ch in range(data.shape[0]):
            signal = data[ch]
            
            if domain == 'device':
                # Device: 더 강한 고주파 노이즈 제거
                from scipy.signal import savgol_filter
                # Savitzky-Golay 필터로 고주파 노이즈 제거
                window_length = min(15, len(signal) // 10)
                if window_length % 2 == 0:
                    window_length += 1
                if window_length >= 3:
                    signal = savgol_filter(signal, window_length, 3)
                    
            else:
                # SNUH: 더 부드러운 필터링
                signal = gaussian_filter1d(signal, sigma=0.8)
            
            filtered_data[ch] = signal
            
        return filtered_data

    def resample_to_length(self, data, target_len):
        """향상된 리샘플링 with anti-aliasing"""
        num_channels, orig_len = data.shape
        if orig_len == target_len:
            return data.astype(np.float32, copy=False)
            
        new_data = np.zeros((num_channels, target_len), dtype=np.float32)
        
        # Anti-aliasing을 위한 pre-filtering
        if target_len < orig_len:
            # Downsampling: low-pass filter 적용
            nyquist_ratio = target_len / orig_len
            cutoff = nyquist_ratio * 0.9  # 90% of Nyquist
            
            for ch in range(num_channels):
                # Simple low-pass with Gaussian
                filtered_signal = gaussian_filter1d(data[ch], sigma=1.0/cutoff)
                
                # Interpolation
                x_old = np.linspace(0, 1, orig_len)
                x_new = np.linspace(0, 1, target_len)
                new_data[ch] = np.interp(x_new, x_old, filtered_signal)
        else:
            # Upsampling: 단순 interpolation
            x_old = np.linspace(0, 1, orig_len)
            x_new = np.linspace(0, 1, target_len)
            for ch in range(num_channels):
                new_data[ch] = np.interp(x_new, x_old, data[ch])
                
        return new_data

    def channel_selection_with_diversity(self, data, domain, seed):
        """다양성을 고려한 채널 선택"""
        if self.num_select is None or self.num_select >= data.shape[0]:
            return data
            
        rng = np.random.RandomState(seed + self.seed_offset)
        num_channels = data.shape[0]
        
        # 채널별 신호 강도 계산
        channel_energies = np.array([np.var(data[ch]) for ch in range(num_channels)])
        
        # 에너지 기준으로 채널을 그룹화
        energy_percentiles = np.percentile(channel_energies, [25, 50, 75])
        
        selected = []
        channels_used = set()
        
        # 각 에너지 그룹에서 균등하게 선택
        for i in range(self.num_select):
            if i % 3 == 0:
                # High energy channels
                candidates = np.where(channel_energies >= energy_percentiles[2])[0]
            elif i % 3 == 1:
                # Medium energy channels  
                candidates = np.where((channel_energies >= energy_percentiles[1]) & 
                                    (channel_energies < energy_percentiles[2]))[0]
            else:
                # Low energy channels
                candidates = np.where(channel_energies < energy_percentiles[1])[0]
            
            # 이미 사용된 채널 제외
            candidates = [c for c in candidates if c not in channels_used]
            
            if len(candidates) == 0:
                # Fall back to any unused channel
                candidates = [c for c in range(num_channels) if c not in channels_used]
            
            if len(candidates) > 0:
                selected_ch = rng.choice(candidates)
                selected.append(data[selected_ch])
                channels_used.add(selected_ch)
            else:
                # 마지막 수단: 임의 채널 재사용
                selected_ch = rng.randint(num_channels)
                selected.append(data[selected_ch])
        
        return np.stack(selected).astype(np.float32)

    def __call__(self, data, domain, seed=42):
        # 1. 노이즈 감소 (선택적)
        if self.noise_reduction:
            data = self.noise_reduction_filter(data, domain)
        
        # 2. 도메인별 envelope 처리
        if domain == 'device':
            # Device: No Envelope
            smoothed = gaussian_filter1d(data, sigma=self.sigma, axis=0)
        else:
            # SNUH: Enhanced Envelope
            envelope_data = self.adaptive_envelope_extraction(data)
            smoothed = gaussian_filter1d(envelope_data, sigma=self.sigma, axis=0)
        
        # 3. 다운샘플링
        downsampled = smoothed[:, ::self.downsample_rate]

        # 4. 도메인별 target_len 설정
        if domain == "SNUH_uncut":
            target_len = 400
        elif domain in ["SNUH", "SNUH_degrade"]:
            target_len = 220
        else:
            target_len = self.default_max_len

        # 5. 길이 조정 (향상된 리샘플링)
        if domain in ["SNUH", "SNUH_degrade", "SNUH_uncut"]:
            processed = self.resample_to_length(downsampled, target_len)
        else:
            if downsampled.shape[1] >= target_len:
                processed = downsampled[:, :target_len]
            else:
                pad = target_len - downsampled.shape[1]
                processed = np.pad(downsampled, ((0,0),(0,pad)), mode="edge")

        # 6. 향상된 정규화
        processed_normalized = np.zeros_like(processed)
        for ch in range(processed.shape[0]):
            ch_data = processed[ch]
            
            # Robust normalization (outlier에 덜 민감)
            ch_median = np.median(ch_data)
            ch_mad = np.median(np.abs(ch_data - ch_median))  # Median Absolute Deviation
            
            if ch_mad > 1e-8:
                # MAD 기반 정규화
                processed_normalized[ch] = (ch_data - ch_median) / (1.4826 * ch_mad)
            else:
                # Fall back to standard normalization
                ch_mean = ch_data.mean()
                ch_std = ch_data.std() + 1e-8
                processed_normalized[ch] = (ch_data - ch_mean) / ch_std

        # 7. 향상된 채널 선택
        if domain in ["SNUH_uncut", "SNUH", "SNUH_degrade"] and self.num_select is not None:
            return self.channel_selection_with_diversity(processed_normalized, domain, seed)
        else:
            return processed_normalized.astype(np.float32)

# 원본 전처리 클래스 (호환성)
class PreprocessTransform:
    """Original preprocessing for backward compatibility"""
    def __init__(self, num_select=None, sigma=1, downsample_rate=10, max_len=220,
                 seed_offset=10, wearable_envelope=False):
        self.num_select = num_select
        self.sigma = sigma
        self.downsample_rate = downsample_rate
        self.default_max_len = max_len
        self.seed_offset = seed_offset
        self.wearable_envelope = bool(wearable_envelope)

    def resample_to_length(self, data, target_len):
        num_channels, orig_len = data.shape
        if orig_len == target_len:
            return data.astype(np.float32, copy=False)
        new_data = np.zeros((num_channels, target_len), dtype=np.float32)
        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, target_len)
        for ch in range(num_channels):
            new_data[ch] = np.interp(x_new, x_old, data[ch])
        return new_data

    def __call__(self, data, domain, seed=42):
        if domain == 'device':
            device_data = (np.abs(hilbert(data, axis=1))
                           if self.wearable_envelope else data)
            smoothed = gaussian_filter1d(device_data, sigma=self.sigma, axis=0)
        else:
            envelope_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                envelope_data[ch] = np.abs(hilbert(data[ch]))
            smoothed = gaussian_filter1d(envelope_data, sigma=self.sigma, axis=0)
        
        downsampled = smoothed[:, ::self.downsample_rate]

        if domain == "SNUH_uncut":
            target_len = 400
        elif domain in ["SNUH", "SNUH_degrade"]:
            target_len = 220
        else:
            target_len = self.default_max_len

        if domain in ["SNUH", "SNUH_degrade", "SNUH_uncut"]:
            processed = self.resample_to_length(downsampled, target_len)
        else:
            if downsampled.shape[1] >= target_len:
                processed = downsampled[:, :target_len]
            else:
                pad = target_len - downsampled.shape[1]
                processed = np.pad(downsampled, ((0,0),(0,pad)), mode="edge")

        processed_normalized = np.zeros_like(processed)
        for ch in range(processed.shape[0]):
            ch_mean = processed[ch].mean()
            ch_std = processed[ch].std() + 1e-8
            processed_normalized[ch] = (processed[ch] - ch_mean) / ch_std

        if domain in ["SNUH_uncut", "SNUH", "SNUH_degrade"] and self.num_select is not None:
            rng = np.random.RandomState(seed + self.seed_offset)
            num_channels = processed_normalized.shape[0]
            group_size = max(1, num_channels // self.num_select)
            selected = []
            for i in range(self.num_select):
                start = i * group_size
                end = start + group_size if i < self.num_select - 1 else num_channels
                col_idx = rng.choice(range(start, end))
                selected.append(processed_normalized[col_idx])
            return np.stack(selected).astype(np.float32)
        else:
            return processed_normalized.astype(np.float32)

# ---------------------- 유틸 함수 ----------------------
def _safe_float_list(str_list):
    out = []
    for x in str_list:
        try: 
            out.append(float(x))
        except: 
            continue
    return out

# ---------------------- 향상된 데이터셋 ----------------------
class FolderDatasetEnhanced(Dataset):
    """
    Enhanced RF 데이터셋 로더
    - 더 정교한 증강 기법
    - 볼륨 기반 샘플링 가중치
    - 향상된 오류 처리
    """
    def __init__(self, dataframe, transform_h=None, transform_s=None, augmentation_factor=1,
                 device_crop=(300, 3000), device_crop_jitter=0, 
                 use_volume_weighting=False, enhanced_augmentation=True):
        self.df = dataframe.reset_index(drop=True)
        self.transform_h = transform_h
        self.transform_s = transform_s
        self.device_crop = device_crop
        self.device_crop_jitter = int(device_crop_jitter) if device_crop_jitter else 0
        self.use_volume_weighting = use_volume_weighting
        self.enhanced_augmentation = enhanced_augmentation

        # 볼륨 기반 가중치 계산
        if use_volume_weighting:
            self.volume_weights = self._calculate_volume_weights()
        
        # 증강 인덱스 생성
        indices = []
        for i, row in self.df.iterrows():
            # 작은 볼륨에 더 많은 증강 적용
            if self.use_volume_weighting and "SNUH" in row['domain']:
                volume = row['volume']
                if volume < 100:
                    factor = augmentation_factor * 2  # 2배 더 증강
                elif volume < 200:
                    factor = int(augmentation_factor * 1.5)
                else:
                    factor = augmentation_factor
            else:
                factor = augmentation_factor if "SNUH" in row['domain'] else 1
            
            indices.extend([i] * factor)
        
        self.indices = np.array(indices, dtype=np.int64)
        
        # 통계 정보 출력
        domain_counts = {}
        for domain in self.df['domain'].unique():
            domain_counts[domain] = np.sum([self.df.iloc[idx]['domain'] == domain for idx in self.indices])
        print(f"Enhanced dataset composition: {domain_counts}")

    def _calculate_volume_weights(self):
        """작은 볼륨에 더 높은 가중치 부여"""
        volumes = self.df['volume'].values
        # Inverse frequency weighting
        volume_bins = np.array([0, 100, 200, 400, 700])
        weights = np.zeros_like(volumes)
        
        for i in range(len(volume_bins)-1):
            mask = (volumes >= volume_bins[i]) & (volumes < volume_bins[i+1])
            count = np.sum(mask)
            if count > 0:
                weights[mask] = 1.0 / count
        
        return weights / weights.max()  # Normalize

    def _enhanced_device_preprocessing(self, data, item_seed):
        """향상된 Device 전처리"""
        # 1. 추가적인 TGC 보정
        data_length = data.shape[1]
        Fs = 20e6; c = 1500.0
        dist = np.arange(data_length, dtype=np.float32) / Fs * c / 2.0 * 1e3
        
        # 깊이별 적응적 TGC
        adaptive_tgc = np.exp(0.1 * dist / 10.0)
        adaptive_tgc[data_length//2:] = np.exp(0.35 * dist[data_length//2:] / 10.0)
        data *= adaptive_tgc[np.newaxis, :]

        # 2. 선택적 주파수 대역 강조
        if self.enhanced_augmentation:
            rng = np.random.RandomState(item_seed ^ 0x12345678)
            # 랜덤하게 특정 주파수 대역 강조
            if rng.random() < 0.3:  # 30% 확률
                from scipy.signal import butter, filtfilt
                
                # 2-3MHz 또는 3-4MHz 대역 중 선택
                if rng.random() < 0.5:
                    low_freq, high_freq = 2.0e6, 3.0e6
                else:
                    low_freq, high_freq = 3.0e6, 4.0e6
                
                b, a = butter(N=5, Wn=[low_freq, high_freq], btype='band', fs=Fs)
                for i in range(data.shape[0]):
                    enhanced_band = filtfilt(b, a, data[i, :])
                    data[i, :] = data[i, :] + 0.2 * enhanced_band  # 20% 강조

        return data

    def _enhanced_snuh_augmentation(self, h_data, s_data, row, item_seed):
        """향상된 SNUH 증강"""
        if not self.enhanced_augmentation:
            return h_data, s_data
        
        rng = np.random.RandomState(item_seed ^ 0x87654321)
        
        # 1. 볼륨 기반 적응적 증강
        volume = row['volume']
        
        if volume < 150:  # 작은 볼륨: 더 강한 증강
            # 시간축 스트레칭/압축
            if rng.random() < 0.4:
                stretch_factor = rng.uniform(0.9, 1.1)
                new_length = int(h_data.shape[1] * stretch_factor)
                
                # 리샘플링
                old_indices = np.linspace(0, h_data.shape[1]-1, h_data.shape[1])
                new_indices = np.linspace(0, h_data.shape[1]-1, new_length)
                
                h_stretched = np.zeros((h_data.shape[0], new_length), dtype=np.float32)
                s_stretched = np.zeros((s_data.shape[0], new_length), dtype=np.float32)
                
                for ch in range(h_data.shape[0]):
                    h_stretched[ch] = np.interp(new_indices, old_indices, h_data[ch])
                for ch in range(s_data.shape[0]):
                    s_stretched[ch] = np.interp(new_indices, old_indices, s_data[ch])
                
                h_data, s_data = h_stretched, s_stretched
        
        # 2. 채널별 진폭 변조
        if rng.random() < 0.3:
            for ch in range(h_data.shape[0]):
                amp_factor = rng.uniform(0.8, 1.2)
                h_data[ch] *= amp_factor
            for ch in range(s_data.shape[0]):
                amp_factor = rng.uniform(0.8, 1.2)
                s_data[ch] *= amp_factor
        
        # 3. 부분적 채널 마스킹 (작은 볼륨만)
        if volume < 100 and rng.random() < 0.2:
            # 랜덤하게 일부 채널에 노이즈 추가
            mask_channels = rng.choice(h_data.shape[0], size=h_data.shape[0]//3, replace=False)
            for ch in mask_channels:
                noise_strength = rng.uniform(0.05, 0.1)
                h_data[ch] += noise_strength * rng.randn(h_data.shape[1])
        
        return h_data, s_data

    def _load_device(self, row, item_seed):
        """향상된 Device 로딩"""
        with open(row['txt_path'], 'r') as f:
            raw_values = f.read().replace("'", "").replace(" ", "").strip().split(',')
            values = np.array(_safe_float_list(raw_values), dtype=np.float32)
        if len(values) < 6: 
            raise ValueError("Not enough numeric data in device file")

        data_length = len(values) // 6
        data = np.zeros((6, data_length), dtype=np.float32)
        for i in range(6):
            data[i] = values[i * data_length:(i + 1) * data_length] / 1700.0

        # 향상된 전처리 적용
        data = self._enhanced_device_preprocessing(data, item_seed)

        # Band-pass filtering
        Fs = 20e6
        b, a = butter(N=7, Wn=[1.5e6, 4.5e6], btype='band', fs=Fs)
        for i in range(6):
            data[i, :] = filtfilt(b, a, data[i, :]).astype(np.float32)

        data = np.clip(data, -0.1, 0.1)

        # 향상된 시간축 증강
        start, end = self.device_crop
        if self.device_crop_jitter > 0:
            rng = np.random.RandomState(item_seed ^ 0x9E3779B1)
            max_shift = min(self.device_crop_jitter, start, data.shape[1] - end)
            shift = rng.randint(-max_shift, max_shift + 1)
            start = max(0, start + shift)
            end = min(data.shape[1], end + shift)
            
            min_length = 500
            if end - start < min_length:
                center = (start + end) // 2
                start = max(0, center - min_length // 2)
                end = min(data.shape[1], start + min_length)
                
        data = data[:, start:end]

        # 7채널 생성
        avg_23 = np.mean(data[1:3, :], axis=0, keepdims=True)
        data = np.concatenate([data[:5, :], avg_23, data[5:, :]], axis=0).astype(np.float32)

        h_data, s_data = data[4:], data[:4]
        return h_data, s_data

    def _load_snuh(self, row, item_seed):
        """향상된 SNUH 로딩"""
        h_file = glob.glob(os.path.join(row['data_path'], f"{row['Upright_H']}*.csv"))
        s_file = glob.glob(os.path.join(row['data_path'], f"{row['Upright_S']}*.csv"))
        
        if not h_file or not s_file:
            raise ValueError(f"CSV files not found for {row['Upright_H']}/{row['Upright_S']}")
            
        h_data = pd.read_csv(h_file[0], header=None, low_memory=False)\
                    .apply(pd.to_numeric, errors='coerce').dropna(axis=1)\
                    .to_numpy().T.astype(np.float32)
        s_data = pd.read_csv(s_file[0], header=None, low_memory=False)\
                    .apply(pd.to_numeric, errors='coerce').dropna(axis=1)\
                    .to_numpy().T.astype(np.float32)
        
        if h_data.size == 0 or s_data.size == 0:
            raise ValueError("Invalid CSV with no numeric data")
        
        # 향상된 증강 적용
        h_data, s_data = self._enhanced_snuh_augmentation(h_data, s_data, row, item_seed)
        
        return h_data, s_data

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        row = self.df.iloc[real_idx]
        
        augment_idx = idx // len(self.df)
        item_seed = ((real_idx * 1315423911) + (augment_idx * 982451653)) & 0xFFFFFFFF

        tries, max_tries = 0, 5
        while True:
            try:
                if row['domain'] == 'device':
                    h_data, s_data = self._load_device(row, item_seed)
                else:
                    h_data, s_data = self._load_snuh(row, item_seed)

                if self.transform_h is not None:
                    h_data = self.transform_h(h_data, domain=row['domain'], seed=item_seed + 11)
                if self.transform_s is not None:
                    s_data = self.transform_s(s_data, domain=row['domain'], seed=item_seed + 17)

                # 향상된 SNUH_degrade 처리
                if row['domain'] == 'SNUH_degrade':
                    N = h_data.shape[1]
                    t = np.linspace(0, 1, N, dtype=np.float32)
                    rng = np.random.RandomState(item_seed ^ 0x85EBCA6B)
                    
                    # 볼륨 기반 적응적 열화
                    volume = row['volume']
                    if volume < 100:
                        decay_strength = rng.uniform(2.0, 3.0)  # 더 강한 열화
                        noise_level = 0.05
                    elif volume < 200:
                        decay_strength = rng.uniform(1.8, 2.5)
                        noise_level = 0.04
                    else:
                        decay_strength = rng.uniform(1.5, 2.2)
                        noise_level = 0.03
                    
                    decay = np.exp(-decay_strength * t).astype(np.float32)
                    
                    noise_h = (noise_level * rng.randn(*h_data.shape)).astype(np.float32)
                    noise_s = (noise_level * rng.randn(*s_data.shape)).astype(np.float32)
                    h_data = h_data * decay[np.newaxis, :] + noise_h
                    s_data = s_data * decay[np.newaxis, :] + noise_s

                if not (np.isfinite(h_data).all() and np.isfinite(s_data).all()):
                    raise ValueError("Non-finite values in processed data")

                return {
                    'horizon':   torch.tensor(h_data, dtype=torch.float32),
                    'sagittal':  torch.tensor(s_data, dtype=torch.float32),
                    'volume_gt': torch.tensor(row['volume'], dtype=torch.float32),
                    'domain':    row['domain']
                }

            except Exception as e:
                tries += 1
                if tries >= max_tries:
                    next_idx = (idx + 1) % len(self.indices)
                    print(f"[WARN] Skipping idx {real_idx} ({row['domain']}) due to error: {e}")
                    return self.__getitem__(next_idx)
                item_seed = (item_seed + 12345) & 0xFFFFFFFF
                continue

# 기존 데이터셋 클래스 (호환성)
class FolderDatasetUnified(Dataset):
    """Original dataset for backward compatibility"""
    def __init__(self, dataframe, transform_h=None, transform_s=None, augmentation_factor=1,
                 device_crop=(300, 3000), device_crop_jitter=0):
        self.df = dataframe.reset_index(drop=True)
        self.transform_h = transform_h
        self.transform_s = transform_s
        self.device_crop = device_crop
        self.device_crop_jitter = int(device_crop_jitter) if device_crop_jitter else 0

        indices = []
        for i, row in self.df.iterrows():
            factor = augmentation_factor if "SNUH" in row['domain'] else 1
            indices.extend([i] * factor)
        self.indices = np.array(indices, dtype=np.int64)
        
        domain_counts = {}
        for domain in self.df['domain'].unique():
            domain_counts[domain] = np.sum([self.df.iloc[idx]['domain'] == domain for idx in self.indices])
        print(f"Dataset composition after augmentation: {domain_counts}")

    def __len__(self): 
        return len(self.indices)

    def _load_device(self, row, item_seed):
        with open(row['txt_path'], 'r') as f:
            raw_values = f.read().replace("'", "").replace(" ", "").strip().split(',')
            values = np.array(_safe_float_list(raw_values), dtype=np.float32)
        if len(values) < 6: 
            raise ValueError("Not enough numeric data in device file")

        data_length = len(values) // 6
        data = np.zeros((6, data_length), dtype=np.float32)
        for i in range(6):
            data[i] = values[i * data_length:(i + 1) * data_length] / 1700.0

        Fs = 20e6; c = 1500.0
        dist = np.arange(data_length, dtype=np.float32) / Fs * c / 2.0 * 1e3
        TGC = np.concatenate([
            np.exp(0.15 * dist[:data_length//2] / 10.0),
            np.exp(0.4  * dist[data_length//2:] / 10.0)
        ]).astype(np.float32)
        data *= TGC[np.newaxis, :]

        b, a = butter(N=7, Wn=[1.5e6, 4.5e6], btype='band', fs=Fs)
        for i in range(6):
            data[i, :] = filtfilt(b, a, data[i, :]).astype(np.float32)

        data = np.clip(data, -0.1, 0.1)

        start, end = self.device_crop
        if self.device_crop_jitter > 0:
            rng = np.random.RandomState(item_seed ^ 0x9E3779B1)
            max_shift = min(self.device_crop_jitter, start, data.shape[1] - end)
            shift = rng.randint(-max_shift, max_shift + 1)
            start = max(0, start + shift)
            end = min(data.shape[1], end + shift)
            
            min_length = 500
            if end - start < min_length:
                center = (start + end) // 2
                start = max(0, center - min_length // 2)
                end = min(data.shape[1], start + min_length)
                
        data = data[:, start:end]

        avg_23 = np.mean(data[1:3, :], axis=0, keepdims=True)
        data = np.concatenate([data[:5, :], avg_23, data[5:, :]], axis=0).astype(np.float32)

        h_data, s_data = data[4:], data[:4]
        return h_data, s_data

    def _load_snuh(self, row):
        h_file = glob.glob(os.path.join(row['data_path'], f"{row['Upright_H']}*.csv"))
        s_file = glob.glob(os.path.join(row['data_path'], f"{row['Upright_S']}*.csv"))
        
        if not h_file or not s_file:
            raise ValueError(f"CSV files not found for {row['Upright_H']}/{row['Upright_S']}")
            
        h_data = pd.read_csv(h_file[0], header=None, low_memory=False)\
                    .apply(pd.to_numeric, errors='coerce').dropna(axis=1)\
                    .to_numpy().T.astype(np.float32)
        s_data = pd.read_csv(s_file[0], header=None, low_memory=False)\
                    .apply(pd.to_numeric, errors='coerce').dropna(axis=1)\
                    .to_numpy().T.astype(np.float32)
        
        if h_data.size == 0 or s_data.size == 0:
            raise ValueError("Invalid CSV with no numeric data")
        return h_data, s_data

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        row = self.df.iloc[real_idx]
        
        augment_idx = idx // len(self.df)
        item_seed = ((real_idx * 1315423911) + (augment_idx * 982451653)) & 0xFFFFFFFF

        tries, max_tries = 0, 5
        while True:
            try:
                if row['domain'] == 'device':
                    h_data, s_data = self._load_device(row, item_seed)
                else:
                    h_data, s_data = self._load_snuh(row)

                if self.transform_h is not None:
                    h_data = self.transform_h(h_data, domain=row['domain'], seed=item_seed + 11)
                if self.transform_s is not None:
                    s_data = self.transform_s(s_data, domain=row['domain'], seed=item_seed + 17)

                if row['domain'] == 'SNUH_degrade':
                    N = h_data.shape[1]
                    t = np.linspace(0, 1, N, dtype=np.float32)
                    rng = np.random.RandomState(item_seed ^ 0x85EBCA6B)
                    decay_strength = rng.uniform(1.5, 2.5)
                    decay = np.exp(-decay_strength * t).astype(np.float32)
                    
                    noise_h = (0.03 * rng.randn(*h_data.shape)).astype(np.float32)
                    noise_s = (0.03 * rng.randn(*s_data.shape)).astype(np.float32)
                    h_data = h_data * decay[np.newaxis, :] + noise_h
                    s_data = s_data * decay[np.newaxis, :] + noise_s

                if not (np.isfinite(h_data).all() and np.isfinite(s_data).all()):
                    raise ValueError("Non-finite values in processed data")

                return {
                    'horizon':   torch.tensor(h_data, dtype=torch.float32),
                    'sagittal':  torch.tensor(s_data, dtype=torch.float32),
                    'volume_gt': torch.tensor(row['volume'], dtype=torch.float32),
                    'domain':    row['domain']
                }

            except Exception as e:
                tries += 1
                if tries >= max_tries:
                    next_idx = (idx + 1) % len(self.indices)
                    print(f"[WARN] Skipping idx {real_idx} ({row['domain']}) due to error: {e}")
                    return self.__getitem__(next_idx)
                item_seed = (item_seed + 12345) & 0xFFFFFFFF
                continue
