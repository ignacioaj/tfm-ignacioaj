from pathlib import Path
import mne
import pandas as pd
import numpy as np

from scripts.signal_preprocessing import raw_to_epochs
from scripts.feature_extraction import extract_features_from_parquet
from scripts.signal_preprocessing import validate_parquet


def main():
    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    parquet_path = output_dir / "epochs.parquet"

    if parquet_path.exists():
        parquet_path.unlink()

    edf_files = sorted(data_dir.glob("*.edf"))
    if not edf_files:
        print("No se encontraron archivos EDF")
        return

    print(f"Procesando {len(edf_files)} archivos EDF")

    # ITERAR SOBRE EDFs (sujetos)
    rows = []

    for subject_idx, edf_path in enumerate(edf_files):
        print(f"\n Sujeto {subject_idx} | Archivo: {edf_path.name}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        epochs = raw_to_epochs(raw)
        if len(epochs) == 0:
            print("No se generaron epochs válidas, se omite")
            continue

        X = epochs.get_data()               # (n_epochs, n_channels, n_times)
        y = epochs.events[:, 2] - 1         # 0 = non-target, 1 = target

        n_epochs, n_channels, n_times = X.shape

        for ep_idx in range(n_epochs):
            rows.append({
                "subject": subject_idx,
                "epoch": ep_idx,
                "label": int(y[ep_idx]),
                "data": [X[ep_idx, ch].astype(float).tolist()
                         for ch in range(n_channels)]
            })

        print(f"   Epochs válidas: {n_epochs} | Targets: {(y == 1).sum()}")

    if not rows:
        print(" No se generaron epochs en ningún sujeto")
        return

    df = pd.DataFrame(rows)

    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        index=False,
        compression="snappy"
    )

    print(f"  Parquet guardado correctamente:")
    print(f"  Ruta: {parquet_path}")
    print(f"  Epochs totales: {len(df)}")
    print(f"  Columnas: {list(df.columns)}")

    print("\nValidando las epochs...")
    validate_parquet(parquet_path)

    print("\nExtrayendo features (temporal / frecuencia / espacial)...")
    extract_features_from_parquet(parquet_path, output_dir)

    print("\n Pipeline completo ejecutado correctamente")
    print("  - Epochs válidas")
    print("  - P300 detectado")
    print("  - Features listas para clasificación")


if __name__ == "__main__":
    main()
