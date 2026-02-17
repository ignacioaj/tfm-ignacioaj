import numpy as np
import pandas as pd
import ast
import os
import mne
from scipy.stats import skew, kurtosis
from scipy.signal import welch, detrend

SF = 256
EPOCH_TMIN = -0.2
P300_WIN = (0.25, 0.6)
P300_CHANNELS = ["EEG_Pz", "EEG_Cz", "EEG_CPz", "EEG_P3", "EEG_P4", "EEG_CP3", "EEG_CP4"]
BANDS = {
    'delta': (1, 4),
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (13, 30)
}

# -------------------------------
# Funciones auxiliares
# -------------------------------
def time_to_idx(t0, t1, sf, epoch_tmin=EPOCH_TMIN):
    i0 = int((t0 - epoch_tmin) * sf)
    i1 = int((t1 - epoch_tmin) * sf)
    return i0, i1

def bandpower(signal, sf, fmin, fmax):
    freqs, psd = welch(signal, fs=sf, nperseg=min(len(signal), sf))
    idx = (freqs >= fmin) & (freqs <= fmax)
    total_power = np.trapz(psd, freqs)
    band_power = np.trapz(psd[idx], freqs[idx])
    return band_power / max(total_power, 1e-12)  # relativa

def extract_temporal_features(epoch, ch_names, sf=SF):
    features = {}
    p0, p1 = time_to_idx(*P300_WIN, sf)
    for ch_idx, ch_name in enumerate(ch_names):
        signal = epoch[ch_idx, p0:p1]
        signal = detrend(signal)
        baseline = np.mean(epoch[ch_idx, :p0])
        signal_rel = signal - baseline
        # Ventanas cortas (50 ms)
        win_len = int(0.05 * sf)
        for w_start in range(0, len(signal_rel) - win_len, win_len):
            win = signal_rel[w_start:w_start + win_len]
            win_center = w_start + win_len // 2
            features[f"{ch_name}_mean_{win_center}"] = np.mean(win)
            features[f"{ch_name}_max_{win_center}"] = np.max(win)
            features[f"{ch_name}_min_{win_center}"] = np.min(win)
            features[f"{ch_name}_auc_{win_center}"] = np.trapz(win, dx=1 / sf)
        # Global
        features[f"{ch_name}_std"] = np.std(signal_rel)
        features[f"{ch_name}_skew"] = skew(signal_rel)
        features[f"{ch_name}_kurtosis"] = kurtosis(signal_rel)
        features[f"{ch_name}_ptp"] = np.ptp(signal_rel)
        features[f"{ch_name}_peak_amp"] = np.max(signal_rel)
        features[f"{ch_name}_peak_lat"] = np.argmax(signal_rel) / sf
    return features

def extract_frequency_features(epoch, ch_names, sf=SF):
    features = {}
    p0, p1 = time_to_idx(*P300_WIN, sf)
    for ch_idx, ch_name in enumerate(ch_names):
        signal = epoch[ch_idx, p0:p1]
        for band_name, (fmin, fmax) in BANDS.items():
            bp = bandpower(signal, sf, fmin, fmax)
            features[f"{ch_name}_bp_{band_name}"] = bp
        # Ratio theta/alpha
        th = features.get(f"{ch_name}_bp_theta", 0)
        al = features.get(f"{ch_name}_bp_alpha", 1e-6)
        features[f"{ch_name}_theta_alpha_ratio"] = th / al
    return features

def extract_spatial_features(epochs_array, labels, ch_names, sf=SF, n_components=6):
    from mne.preprocessing import Xdawn
    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=['eeg'] * len(ch_names))
    n_epochs = epochs_array.shape[0]
    events = np.zeros((n_epochs, 3), int)
    events[:, 0] = np.arange(n_epochs)
    events[:, 2] = labels + 1
    epochs_mne = mne.EpochsArray(epochs_array, info, events=events, tmin=EPOCH_TMIN, verbose=False)
    xd = Xdawn(n_components=n_components)
    xd.fit(epochs_mne, epochs_mne.events[:, 2])
    epochs_xd = xd.transform(epochs_mne)
    features_list = []
    for epoch_idx in range(n_epochs):
        feats = {}
        xd_epoch = epochs_xd[epoch_idx]
        for comp in range(xd_epoch.shape[0]):
            sig = xd_epoch[comp]
            feats[f"xd_comp{comp}_mean"] = np.mean(sig)
            feats[f"xd_comp{comp}_max"] = np.max(sig)
            feats[f"xd_comp{comp}_min"] = np.min(sig)
            feats[f"xd_comp{comp}_auc"] = np.trapz(sig, dx=1 / sf)
        features_list.append(feats)
    return features_list

# -------------------------------
# Función principal
# -------------------------------
def extract_features_from_parquet(parquet_path, output_dir):
    df_epochs = pd.read_parquet(parquet_path)
    os.makedirs(output_dir, exist_ok=True)

    all_epochs = []
    labels = []
    subjects = []

    for _, row in df_epochs.iterrows():
        epoch = ast.literal_eval(row['data']) if isinstance(row['data'], str) else row['data']
        epoch = [np.asarray(ch, dtype=float) for ch in epoch]
        epoch = np.stack(epoch[:len(P300_CHANNELS)], axis=0)
        all_epochs.append(epoch)
        labels.append(row['label'])
        subjects.append(row['subject'])

    all_epochs = np.stack(all_epochs, axis=0)
    labels = np.array(labels)
    subjects = np.array(subjects)

    # ====== Normalizar por sujeto ======
    for subj in np.unique(subjects):
        idx = subjects == subj
        mean_s = all_epochs[idx].mean(axis=(1,2), keepdims=True)
        std_s  = all_epochs[idx].std(axis=(1,2), keepdims=True)
        all_epochs[idx] = (all_epochs[idx] - mean_s) / (std_s + 1e-12)

    # ====== Temporal ======
    rows_temp = []
    for i in range(all_epochs.shape[0]):
        feats = extract_temporal_features(all_epochs[i], P300_CHANNELS)
        feats.update({"subject": subjects[i], "label": labels[i]})
        rows_temp.append(feats)
    df_temp = pd.DataFrame(rows_temp)
    # Eliminar features constantes
    df_temp = df_temp.loc[:, df_temp.nunique() > 1]
    df_temp.to_parquet(os.path.join(output_dir, "features_temporal_clean.parquet"), index=False)

    # ====== Frecuencia ======
    rows_freq = []
    for i in range(all_epochs.shape[0]):
        feats = extract_frequency_features(all_epochs[i], P300_CHANNELS)
        feats.update({"subject": subjects[i], "label": labels[i]})
        rows_freq.append(feats)
    df_freq = pd.DataFrame(rows_freq)
    df_freq = df_freq.loc[:, df_freq.nunique() > 1]
    df_freq.to_parquet(os.path.join(output_dir, "features_frequency_clean.parquet"), index=False)

    # ====== Espacial ======
    rows_spa = extract_spatial_features(all_epochs, labels, P300_CHANNELS)
    for i, feats in enumerate(rows_spa):
        feats.update({"subject": subjects[i], "label": labels[i]})
    df_spa = pd.DataFrame(rows_spa)
    df_spa = df_spa.loc[:, df_spa.nunique() > 1]
    df_spa.to_parquet(os.path.join(output_dir, "features_spatial_clean.parquet"), index=False)

    print("✅ Features extraídas y limpiadas (constantes eliminadas, normalización por sujeto)")
