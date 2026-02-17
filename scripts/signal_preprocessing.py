import numpy as np
import mne
import pandas as pd
import os

def preprocess_raw(raw,
                         l_freq=0.1,
                         h_freq=30,
                         notch_freq=50):
    raw.notch_filter(notch_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    raw.set_eeg_reference('average', projection=True)
    return raw

def extract_events_from_bigp3bci(raw):
    stim_begin = raw.copy().pick_channels(['StimulusBegin']).get_data()[0]
    stim_type = raw.copy().pick_channels(['StimulusType']).get_data()[0]

    onsets = np.where((np.diff(stim_begin) > 0) & (stim_begin[1:] == 1))[0] + 1

    events = np.zeros((len(onsets), 3), dtype=int)
    events[:, 0] = onsets

    labels = stim_type[onsets].astype(int)
    events[:, 2] = labels + 1  # 1 = non-target, 2 = target

    event_id = {'non_target': 1, 'target': 2}
    return events, event_id

def raw_to_epochs(raw,
                       tmin=-0.2,
                       tmax=0.6,
                       l_freq=0.1,
                       h_freq=30):

    eeg_channels = [ch for ch in raw.ch_names if ch.startswith("EEG_")]
    event_channels = ['StimulusBegin', 'StimulusType']

    raw.pick_channels(eeg_channels + event_channels)

    events, event_id = extract_events_from_bigp3bci(raw)
    raw.drop_channels(event_channels)

    raw = preprocess_raw(raw, l_freq, h_freq)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        preload=True,
        reject=dict(eeg=200e-6),
        reject_by_annotation=True,
        verbose=False
    )

    return epochs

def epochs_to_parquet(epochs, subject_id, out_path):
    rows = []
    for i, ep in enumerate(epochs):
        rows.append({
            "subject": subject_id,
            "epoch": i,
            "label": epochs.events[i, 2] - 1,
            "data": ep.tolist()
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df

def validate_parquet(parquet_path, sf=256, channel_idx=0):
    """
    Validación robusta del P300 usando un solo canal.
    Evita errores por estructuras irregulares en el parquet.
    """

    df = pd.read_parquet(parquet_path)

    labels = df["label"].values
    n_epochs = len(df)

    n_targets = np.sum(labels == 1)
    n_nontargets = np.sum(labels == 0)

    assert n_epochs > 0, "No hay epochs"
    assert n_targets > 0, "No hay targets"
    assert n_nontargets > 0, "No hay non-targets"

    # Extraer SOLO un canal (ej: Pz)
    signals = []
    for x in df["data"]:
        ch = np.asarray(x[channel_idx], dtype=float)
        signals.append(ch)

    # Asegurar longitudes consistentes
    min_len = min(len(sig) for sig in signals)
    signals = np.stack([sig[:min_len] for sig in signals])

    tgt = signals[labels == 1]
    nont = signals[labels == 0]

    # Asegurar 2D
    if tgt.ndim == 1:
        tgt = tgt[np.newaxis, :]
    if nont.ndim == 1:
        nont = nont[np.newaxis, :]

    tgt_mean = tgt.mean(axis=0)
    nont_mean = nont.mean(axis=0)

    # Ventana P300
    p0 = int((0.25 + 0.2) * sf)
    p1 = int((0.6 + 0.2) * sf)

    amp_tgt = tgt_mean[p0:p1].mean()
    amp_nont = nont_mean[p0:p1].mean()

    # -----------------------
    print("\n===== VALIDACIÓN P300 (canal único) =====")
    print(f"Epochs totales      : {n_epochs}")
    print(f"Targets             : {n_targets}")
    print(f"Non-targets         : {n_nontargets}")
    print(f"Amplitud Target     : {amp_tgt:.3f}")
    print(f"Amplitud Non-Target : {amp_nont:.3f}")

    if amp_tgt > amp_nont:
        print("VALIDACIÓN CORRECTA: P300 detectado")
    else:
        print("VALIDACIÓN DÉBIL: revisar alineación o eventos")

