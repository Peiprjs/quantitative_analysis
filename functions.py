"""
functions.py
============
Reusable helper functions for the DACIL-WESENSE quantitative analysis pipeline.

This module provides utilities for:
- Directory parsing and file discovery
- Data loading (CSV telemetry, BDF ECG)
- Data synchronization and cleaning
- Exploratory data analysis (EDA)
- Unsupervised machine learning (PCA, K-Means, DBSCAN)
- Output export

All functions follow PEP 8, include NumPy-style docstrings, and use type hints.
"""

import logging
import os
import pickle
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython
from matplotlib.figure import Figure
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Use non-interactive backend by default for batch mode.
# In notebooks/IPython sessions keep the active backend so figures can be shown.
if get_ipython() is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

mne.set_log_level("WARNING")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

_USE_GPU: bool = False


def configure(*, use_gpu: bool = False) -> None:
    """Set runtime options for the functions module.

    Parameters
    ----------
    use_gpu : bool
        If ``True``, attempt to use CuPy for GPU-accelerated signal
        processing (e.g. ECG autocorrelation).  Silently falls back to
        NumPy when CuPy is not installed or no CUDA device is available.
    """
    global _USE_GPU
    _USE_GPU = use_gpu
    if use_gpu:
        try:
            import cupy as cp  # noqa: F401
            logger.info("GPU mode enabled (CuPy %s).", cp.__version__)
        except ImportError:
            logger.warning(
                "configure(use_gpu=True) requested but cupy is not installed; "
                "falling back to CPU."
            )
            _USE_GPU = False


# ---------------------------------------------------------------------------
# Stage annotation helpers
# ---------------------------------------------------------------------------

STAGE_ORDER: List[str] = ["Opwarmen", "Test", "VT1", "VT2", "Herstel"]

# ---------------------------------------------------------------------------
# 1. Directory parsing
# ---------------------------------------------------------------------------


def discover_patient_folders(root_dir: str) -> List[Path]:
    """Return all immediate sub-directories of *root_dir*.

    Parameters
    ----------
    root_dir : str
        Path to the root data directory that contains one sub-folder per
        patient/trial.

    Returns
    -------
    List[Path]
        Sorted list of patient folder paths.
    """
    root = Path(root_dir)
    folders = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    logger.info("Discovered %d patient folder(s) under '%s'.", len(folders), root_dir)
    return folders


def find_bdf_files(folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the L1 and L2 BDF ECG files inside *folder*.

    The expected naming pattern is::

        WESENSETEST_<id>_L1_ECG*.bdf
        WESENSETEST_<id>_L2_ECG*.bdf

    Parameters
    ----------
    folder : Path
        Patient folder to search.

    Returns
    -------
    Tuple[Optional[Path], Optional[Path]]
        (path_to_L1, path_to_L2).  Either may be ``None`` when not found.
    """
    l1_path: Optional[Path] = None
    l2_path: Optional[Path] = None

    for bdf in folder.glob("*.bdf"):
        name = bdf.name.upper()
        if "_L1_ECG" in name:
            l1_path = bdf
        elif "_L2_ECG" in name:
            l2_path = bdf

    return l1_path, l2_path


def find_csv_file(folder: Path) -> Optional[Path]:
    """Return the first ``.csv`` file found in *folder*.

    Parameters
    ----------
    folder : Path
        Patient folder to search.

    Returns
    -------
    Optional[Path]
        Path to the CSV file, or ``None`` if not found.
    """
    files = list(folder.glob("*.csv"))
    return files[0] if files else None


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------


def load_telemetry(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load patient metadata and telemetry data from a WESENSE CPET CSV file.

    WESENSE exports begin with a 5-row metadata header (patient info), followed
    by the column-name row, a units row, and then the time-series data.  This
    function reads both sections separately.

    Parameters
    ----------
    csv_path : Path
        Path to the ``.csv`` file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(info_df, telemetry_df)`` where *info_df* contains the raw metadata
        rows (columns are integer-indexed, no header) and *telemetry_df* has
        named metric columns plus a ``Stage`` column.
    """
    # ── Metadata (rows 0-3) ───────────────────────────────────────────────
    info_df = pd.read_csv(
        csv_path, header=None, nrows=4, encoding_errors="replace"
    )

    # ── Telemetry ─────────────────────────────────────────────────────────
    # Skip rows 0-4 (metadata + blank); row 5 becomes the header.
    # The units row (original row 6) ends up as the first data row; it
    # contains strings in numeric columns and is dropped by _parse_telemetry.
    raw = pd.read_csv(
        csv_path,
        skiprows=list(range(5)),
        header=0,
        encoding_errors="replace",
    )

    logger.debug("CSV rows loaded: %d", len(raw))
    telemetry_df = _parse_telemetry(raw)

    return info_df, telemetry_df


def _parse_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw telemetry DataFrame and add a ``Stage`` column.

    Rows that contain a stage annotation keyword (see :data:`STAGE_ORDER`) are
    used to label subsequent rows until the next annotation is encountered.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as read directly from Excel.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an additional ``Stage`` string column.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Build a lower-case string representation of every row in one pass.
    row_text = df.astype(str).agg(" ".join, axis=1).str.lower()

    # Assign stage labels to annotation rows.  Iterate stages in reverse so
    # that earlier entries in STAGE_ORDER take priority (overwrite later ones).
    stage_series = pd.Series(pd.NA, index=df.index, dtype="object")
    for stage in reversed(STAGE_ORDER):
        mask = row_text.str.contains(stage.lower(), regex=False, na=False)
        stage_series[mask] = stage

    # Forward-fill so every data row inherits the most recent stage label.
    df["Stage"] = stage_series.ffill()

    # Keep only rows that look like data rows:
    #   - first column matches a HH:MM time string (excludes the units row)
    #   - no stage keyword was detected in the row itself (excludes annotation rows)
    annotation_mask = stage_series.notna()
    time_mask = (
        df.iloc[:, 0].astype(str).str.strip().str.match(r"^\d+:\d{2}", na=False)
    )
    df = df[time_mask & ~annotation_mask].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def load_ecg(bdf_path: Path) -> Optional[mne.io.BaseRaw]:
    """Load a BDF ECG file with MNE and return the Raw object.

    Parameters
    ----------
    bdf_path : Path
        Path to the ``.bdf`` file.

    Returns
    -------
    Optional[mne.io.BaseRaw]
        MNE Raw object, or ``None`` if loading fails.
    """
    try:
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=False, verbose=False)
        logger.info("Loaded BDF: %s  (%.1f s)", bdf_path.name, raw.times[-1])
        return raw
    except (ValueError, OSError, RuntimeError) as exc:
        logger.warning("Could not load BDF '%s': %s", bdf_path, exc)
        return None


def extract_ecg_features(
    raw: mne.io.BaseRaw,
    chunk_duration: float = 120.0,
) -> pd.DataFrame:
    """Compute basic ECG summary statistics from a Raw object.

    The recording is processed in *chunk_duration*-second windows, with chunk
    boundaries snapped to BDF data-record boundaries to avoid misaligned reads
    on files with mixed per-channel sampling rates.  Per-channel statistics are
    aggregated across all chunks.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded MNE Raw object (``preload=False`` is fine).
    chunk_duration : float
        Requested window length in seconds (default 120 s).  The actual chunk
        size is rounded up to the nearest BDF record boundary.

    Returns
    -------
    pd.DataFrame
        One row per ECG channel with columns ``channel``, ``mean_abs_uV``,
        ``estimated_hr_bpm``.
    """
    sfreq = raw.info["sfreq"]
    total_samples = raw.n_times

    # ── Determine record-boundary size from BDF metadata ─────────────────
    record_samples = 1
    if hasattr(raw, "_raw_extras") and raw._raw_extras:
        extras = raw._raw_extras[0]
        rec_len = float(extras.get("record_length", [1.0])[0])
        record_samples = max(1, int(round(rec_len * sfreq)))

    # Snap requested chunk to the nearest multiple of record_samples so that
    # every start/stop boundary falls on a BDF record edge.
    chunk_req = max(record_samples, int(chunk_duration * sfreq))
    chunk_samples = int(round(chunk_req / record_samples)) * record_samples

    # ── Channel selection ─────────────────────────────────────────────────
    # Priority: MNE-typed ECG > channels named 'ecg*' > MNE-typed EEG > first 2
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    if len(ecg_picks) == 0:
        # Name-based fallback: pick channels whose label contains 'ecg'
        ecg_picks = np.array(
            [i for i, ch in enumerate(raw.ch_names) if "ecg" in ch.lower()],
            dtype=int,
        )
    if len(ecg_picks) == 0:
        ecg_picks = mne.pick_types(raw.info, eeg=True, meg=False)
    if len(ecg_picks) == 0:
        ecg_picks = np.arange(min(2, len(raw.ch_names)))

    # ── Per-chunk accumulation ─────────────────────────────────────────────
    ch_stats: Dict[int, Dict[str, list]] = {
        idx: {"mean_abs_uV": [], "estimated_hr_bpm": []} for idx in ecg_picks
    }

    start = 0
    while start < total_samples:
        # Snap stop to a record boundary; honour end-of-file on final chunk.
        stop_aligned = start + chunk_samples
        stop = min(stop_aligned, total_samples)
        for ch_idx in ecg_picks:
            data, _ = raw[ch_idx, start:stop]
            data = data.squeeze()
            ch_stats[ch_idx]["mean_abs_uV"].append(
                float(np.mean(np.abs(data)) * 1e6)
            )
            ch_stats[ch_idx]["estimated_hr_bpm"].append(
                _estimate_hr_from_signal(data, sfreq)
            )
        start = stop_aligned  # advance by full aligned chunk, not clipped stop

    records = []
    for ch_idx in ecg_picks:
        hr_vals = [v for v in ch_stats[ch_idx]["estimated_hr_bpm"] if not np.isnan(v)]
        records.append(
            {
                "channel": raw.ch_names[ch_idx],
                "mean_abs_uV": float(np.mean(ch_stats[ch_idx]["mean_abs_uV"])),
                "estimated_hr_bpm": float(np.median(hr_vals)) if hr_vals else float("nan"),
            }
        )

    return pd.DataFrame(records)


def extract_ecg_timeseries(
    raw: mne.io.BaseRaw,
    window_duration: float = 30.0,
) -> pd.DataFrame:
    """Extract time-resolved ECG features using a sliding window.

    For each ECG channel the full signal is R-peak detected, then divided into
    non-overlapping windows.  Per-window statistics include heart rate,
    time-domain HRV (RMSSD, SDNN) and frequency-domain HRV (LF, HF, LF/HF),
    plus an ECG-derived respiration (EDR) breathing rate estimate.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded MNE Raw object (``preload=False`` is fine).
    window_duration : float
        Duration of each analysis window in seconds (default 30 s).
        Frequency-domain HRV values are exploratory at short windows; the
        clinical standard is ≥ 300 s.

    Returns
    -------
    pd.DataFrame
        Columns: ``time_s``, ``channel``, ``hr_bpm``, ``rmssd_ms``,
        ``sdnn_ms``, ``lf_ms2``, ``hf_ms2``, ``lf_hf_ratio``,
        ``breathing_rate_bpm``.
        ``time_s`` is the centre of each window in seconds from recording
        start.  Values are ``NaN`` when fewer than 10 R-peaks are detected
        in a window.
    """
    sfreq = raw.info["sfreq"]
    total_samples = raw.n_times
    window_samples = max(1, int(window_duration * sfreq))

    # Same channel-selection priority as extract_ecg_features.
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    if len(ecg_picks) == 0:
        ecg_picks = np.array(
            [i for i, ch in enumerate(raw.ch_names) if "ecg" in ch.lower()],
            dtype=int,
        )
    if len(ecg_picks) == 0:
        ecg_picks = mne.pick_types(raw.info, eeg=True, meg=False)
    if len(ecg_picks) == 0:
        ecg_picks = np.arange(min(2, len(raw.ch_names)))

    rows: List[Dict] = []
    nan = float("nan")

    for ch_idx in ecg_picks:
        ch_name = raw.ch_names[ch_idx]
        full_data, _ = raw[ch_idx, :]
        full_signal = full_data.squeeze()

        # Detect R-peaks on the entire channel signal once.
        all_r_peaks = _detect_r_peaks(full_signal, sfreq)

        for win_start in range(0, total_samples, window_samples):
            win_stop = min(win_start + window_samples, total_samples)
            centre_s = (win_start + win_stop) / 2.0 / sfreq

            # R-peaks that fall within this window.
            mask = (all_r_peaks >= win_start) & (all_r_peaks < win_stop)
            win_peaks = all_r_peaks[mask]

            if len(win_peaks) < 10:
                rows.append(
                    dict(
                        time_s=centre_s, channel=ch_name,
                        hr_bpm=nan, rmssd_ms=nan, sdnn_ms=nan,
                        lf_ms2=nan, hf_ms2=nan, lf_hf_ratio=nan,
                        breathing_rate_bpm=nan,
                    )
                )
                continue

            # RR intervals in milliseconds.
            rr_ms = np.diff(win_peaks) / sfreq * 1000.0

            # Instantaneous HR from median RR.
            median_rr_s = float(np.median(rr_ms)) / 1000.0
            hr_bpm = 60.0 / median_rr_s if median_rr_s > 0 else nan

            hrv = _compute_hrv_metrics(rr_ms)

            # EDR breathing rate from R-peak amplitudes.
            win_amps = full_signal[win_peaks]
            br_bpm = _estimate_breathing_rate_from_ecg(win_peaks, win_amps, sfreq)

            rows.append(
                dict(
                    time_s=centre_s,
                    channel=ch_name,
                    hr_bpm=hr_bpm,
                    rmssd_ms=hrv["rmssd_ms"],
                    sdnn_ms=hrv["sdnn_ms"],
                    lf_ms2=hrv["lf_ms2"],
                    hf_ms2=hrv["hf_ms2"],
                    lf_hf_ratio=hrv["lf_hf_ratio"],
                    breathing_rate_bpm=br_bpm,
                )
            )

    return pd.DataFrame(rows)


def extract_ecg_timeseries_array(
    data: np.ndarray,
    sfreq: float,
    channel_names: Optional[List[str]] = None,
    window_duration: float = 30.0,
) -> pd.DataFrame:
    """Extract time-resolved ECG features from a raw NumPy array.

    Equivalent to :func:`extract_ecg_timeseries` but accepts a plain NumPy
    array instead of an MNE Raw object, making it compatible with any BDF
    loader (e.g. *pybdf*).

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape ``(n_channels, n_samples)`` containing the ECG
        signal in physical units (e.g. µV or mV).
    sfreq : float
        Sampling frequency in Hz.
    channel_names : list of str, optional
        Labels for each row of *data*.  Defaults to ``["ch0", "ch1", …]``.
    window_duration : float
        Duration of each analysis window in seconds (default 30 s).

    Returns
    -------
    pd.DataFrame
        Same columns as :func:`extract_ecg_timeseries`:
        ``time_s``, ``channel``, ``hr_bpm``, ``rmssd_ms``, ``sdnn_ms``,
        ``lf_ms2``, ``hf_ms2``, ``lf_hf_ratio``, ``breathing_rate_bpm``.
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_channels, total_samples = data.shape
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    window_samples = max(1, int(window_duration * sfreq))
    rows: List[Dict] = []
    nan = float("nan")

    for ch_idx in range(n_channels):
        ch_name = channel_names[ch_idx]
        full_signal = data[ch_idx].astype(float)

        all_r_peaks = _detect_r_peaks(full_signal, sfreq)

        for win_start in range(0, total_samples, window_samples):
            win_stop = min(win_start + window_samples, total_samples)
            centre_s = (win_start + win_stop) / 2.0 / sfreq

            mask = (all_r_peaks >= win_start) & (all_r_peaks < win_stop)
            win_peaks = all_r_peaks[mask]

            if len(win_peaks) < 10:
                rows.append(
                    dict(
                        time_s=centre_s, channel=ch_name,
                        hr_bpm=nan, rmssd_ms=nan, sdnn_ms=nan,
                        lf_ms2=nan, hf_ms2=nan, lf_hf_ratio=nan,
                        breathing_rate_bpm=nan,
                    )
                )
                continue

            rr_ms = np.diff(win_peaks) / sfreq * 1000.0
            median_rr_s = float(np.median(rr_ms)) / 1000.0
            hr_bpm = 60.0 / median_rr_s if median_rr_s > 0 else nan

            hrv = _compute_hrv_metrics(rr_ms)
            win_amps = full_signal[win_peaks]
            br_bpm = _estimate_breathing_rate_from_ecg(win_peaks, win_amps, sfreq)

            rows.append(
                dict(
                    time_s=centre_s,
                    channel=ch_name,
                    hr_bpm=hr_bpm,
                    rmssd_ms=hrv["rmssd_ms"],
                    sdnn_ms=hrv["sdnn_ms"],
                    lf_ms2=hrv["lf_ms2"],
                    hf_ms2=hrv["hf_ms2"],
                    lf_hf_ratio=hrv["lf_hf_ratio"],
                    breathing_rate_bpm=br_bpm,
                )
            )

    return pd.DataFrame(rows)


def save_ecg_checkpoint(
    features: pd.DataFrame, patient_out: Path, label: str
) -> None:
    """Persist ECG feature DataFrame to a pickle checkpoint file.

    Parameters
    ----------
    features : pd.DataFrame
        Output from :func:`extract_ecg_features`.
    patient_out : Path
        Per-patient output directory (created if it does not exist).
    label : str
        BDF label, e.g. ``"L1"`` or ``"L2"``.
    """
    patient_out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = patient_out / f"ecg_checkpoint_{label}.pkl"
    with open(checkpoint_path, "wb") as fh:
        pickle.dump(features, fh)
    logger.info("ECG checkpoint saved: %s", checkpoint_path)


def load_ecg_checkpoint(
    patient_out: Path, label: str
) -> Optional[pd.DataFrame]:
    """Load a pickled ECG feature DataFrame from a checkpoint file.

    Parameters
    ----------
    patient_out : Path
        Per-patient output directory.
    label : str
        BDF label, e.g. ``"L1"`` or ``"L2"``.

    Returns
    -------
    Optional[pd.DataFrame]
        The cached feature DataFrame, or ``None`` if no checkpoint exists.
    """
    checkpoint_path = patient_out / f"ecg_checkpoint_{label}.pkl"
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "rb") as fh:
        features = pickle.load(fh)
    logger.info("ECG checkpoint loaded: %s", checkpoint_path)
    return features


def _estimate_hr_from_signal(signal: np.ndarray, sfreq: float) -> float:
    """Estimate heart rate from a 1-D ECG signal via autocorrelation.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG amplitude array.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Estimated heart rate in beats per minute, or ``float('nan')`` on
        failure.
    """
    try:
        # Band-pass rough filtering using diff as a quick proxy.
        diff_signal = np.diff(signal)

        # Compute autocorrelation via FFT: O(n log n) vs O(n²) for np.correlate.
        if _USE_GPU:
            try:
                import cupy as cp
                sig_gpu = cp.asarray(diff_signal)
                n = len(sig_gpu)
                fft_sig = cp.fft.rfft(sig_gpu, n=2 * n)
                acorr = cp.asnumpy(cp.fft.irfft(fft_sig * cp.conj(fft_sig)))[:n]
            except Exception:
                # Fall back to CPU if CuPy fails at runtime.
                n = len(diff_signal)
                fft_sig = np.fft.rfft(diff_signal, n=2 * n)
                acorr = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n]
        else:
            n = len(diff_signal)
            fft_sig = np.fft.rfft(diff_signal, n=2 * n)
            acorr = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n]

        # Search for the first peak beyond 0.3 s (200 bpm max).
        min_lag = int(0.3 * sfreq)
        max_lag = int(2.0 * sfreq)  # 30 bpm min
        search_region = acorr[min_lag:max_lag]
        if len(search_region) == 0:
            return float("nan")
        peak_idx = int(np.argmax(search_region)) + min_lag
        period_s = peak_idx / sfreq
        return 60.0 / period_s if period_s > 0 else float("nan")
    except (ValueError, IndexError, ZeroDivisionError):
        return float("nan")


def _detect_r_peaks(signal_1d: np.ndarray, sfreq: float) -> np.ndarray:
    """Detect R-peak positions in a 1-D ECG signal.

    Parameters
    ----------
    signal_1d : np.ndarray
        1-D ECG amplitude array (any unit).
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Integer array of sample indices where R-peaks were detected.
    """
    try:
        # Band-pass 5–40 Hz to isolate the QRS complex.
        low, high = 5.0 / (0.5 * sfreq), 40.0 / (0.5 * sfreq)
        low = max(1e-4, min(low, 0.999))
        high = max(low + 1e-4, min(high, 0.999))
        b, a = sp_signal.butter(2, [low, high], btype="band")
        filtered = sp_signal.filtfilt(b, a, signal_1d)

        # Rectify and detect peaks.
        min_dist = int(0.3 * sfreq)  # 200 bpm upper limit
        threshold = 0.3 * float(np.max(np.abs(filtered)))
        peaks, _ = sp_signal.find_peaks(
            np.abs(filtered), distance=min_dist, height=threshold
        )
        return peaks
    except Exception:
        return np.array([], dtype=int)


def _compute_hrv_metrics(rr_ms: np.ndarray) -> Dict[str, float]:
    """Compute time- and frequency-domain HRV metrics from RR intervals.

    Parameters
    ----------
    rr_ms : np.ndarray
        Array of successive RR intervals in milliseconds.

    Returns
    -------
    Dict[str, float]
        Keys: ``sdnn_ms``, ``rmssd_ms``, ``lf_ms2``, ``hf_ms2``,
        ``lf_hf_ratio``.  Values are ``float('nan')`` when the input is
        too short for reliable estimation.
    """
    nan = float("nan")
    result: Dict[str, float] = dict(
        sdnn_ms=nan, rmssd_ms=nan, lf_ms2=nan, hf_ms2=nan, lf_hf_ratio=nan
    )

    if len(rr_ms) < 3:
        return result

    # Time-domain metrics.
    result["sdnn_ms"] = float(np.std(rr_ms, ddof=1))
    successive_diff = np.diff(rr_ms)
    result["rmssd_ms"] = float(np.sqrt(np.mean(successive_diff ** 2)))

    # Frequency-domain: resample RR to 4 Hz for spectral analysis.
    if len(rr_ms) < 5:
        return result

    try:
        rr_s = rr_ms / 1000.0
        cumulative_t = np.cumsum(rr_s)
        cumulative_t -= cumulative_t[0]  # start at 0
        target_fs = 4.0
        t_uniform = np.arange(0, cumulative_t[-1], 1.0 / target_fs)
        if len(t_uniform) < 8:
            return result

        interpolator = interp1d(cumulative_t, rr_ms, kind="linear", fill_value="extrapolate")
        rr_uniform = interpolator(t_uniform)

        freqs, psd = sp_signal.welch(rr_uniform, fs=target_fs, nperseg=min(len(rr_uniform), 64))

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        lf = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else nan
        hf = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else nan

        result["lf_ms2"] = lf
        result["hf_ms2"] = hf
        if not np.isnan(lf) and not np.isnan(hf) and hf > 0:
            result["lf_hf_ratio"] = lf / hf
    except Exception:
        pass

    return result


def _estimate_breathing_rate_from_ecg(
    r_peak_indices: np.ndarray,
    r_amplitudes: np.ndarray,
    sfreq: float,
) -> float:
    """Estimate breathing rate from ECG-derived respiration (EDR).

    Uses the amplitude modulation of R-peaks (a well-established proxy for
    respiratory effort) to extract the dominant breathing frequency.

    Parameters
    ----------
    r_peak_indices : np.ndarray
        Sample indices of detected R-peaks.
    r_amplitudes : np.ndarray
        Signal amplitude at each R-peak (same length as *r_peak_indices*).
    sfreq : float
        Sampling frequency of the original ECG signal in Hz.

    Returns
    -------
    float
        Estimated breathing rate in breaths per minute, or ``float('nan')``
        on failure.
    """
    if len(r_peak_indices) < 8:
        return float("nan")

    try:
        # Times of each R-peak in seconds.
        r_times = r_peak_indices / sfreq

        # Resample amplitude envelope to 4 Hz.
        target_fs = 4.0
        t_uniform = np.arange(r_times[0], r_times[-1], 1.0 / target_fs)
        if len(t_uniform) < 8:
            return float("nan")

        interp = interp1d(r_times, r_amplitudes, kind="linear", fill_value="extrapolate")
        env_uniform = interp(t_uniform)

        # Welch PSD; look for dominant peak in the breathing band (0.1–0.5 Hz = 6–30 bpm).
        freqs, psd = sp_signal.welch(env_uniform, fs=target_fs, nperseg=min(len(env_uniform), 64))
        breath_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not breath_mask.any():
            return float("nan")

        dominant_idx = int(np.argmax(psd[breath_mask]))
        dominant_freq = freqs[breath_mask][dominant_idx]
        return float(dominant_freq * 60.0)
    except Exception:
        return float("nan")


def _bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Band-pass filter a 1D signal with zero-phase SOS filtering.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Low cut-off frequency in Hz.
    highcut : float
        High cut-off frequency in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    sig = np.asarray(signal, dtype=float).squeeze()
    if sig.ndim != 1:
        raise ValueError("Expected a 1D signal")
    if sig.size < 3:
        return sig.copy()
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    nyquist = fs / 2.0
    low = max(float(lowcut), 1e-6)
    high = min(float(highcut), nyquist * 0.99)
    if low >= high:
        return sig.copy()

    sos = sp_signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, sig)


def _notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Apply a notch filter for electrical mains interference.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    fs : float
        Sampling frequency in Hz.
    freq : float
        Interference frequency in Hz (e.g., 50 or 60 Hz).
    quality : float
        Quality factor of the notch filter.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    sig = np.asarray(signal, dtype=float).squeeze()
    if sig.ndim != 1:
        raise ValueError("Expected a 1D signal")
    if sig.size < 3 or fs <= 0 or freq <= 0:
        return sig.copy()

    nyquist = fs / 2.0
    if freq >= nyquist:
        return sig.copy()

    b, a = sp_signal.iirnotch(w0=freq, Q=quality, fs=fs)
    sos = sp_signal.tf2sos(b, a)
    return sp_signal.sosfiltfilt(sos, sig)


def filter_ecg(signal: np.ndarray, fs: float, mains_hz: float = 50.0) -> np.ndarray:
    """Filter ECG with mains notch and 0.5–40 Hz band-pass.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal.
    fs : float
        Sampling frequency in Hz.
    mains_hz : float
        Electrical mains frequency for notch filtering (typically 50 or 60 Hz).

    Returns
    -------
    np.ndarray
        Filtered ECG signal.
    """
    filtered = _notch_filter(signal, fs, freq=mains_hz)
    return _bandpass_filter(filtered, fs, lowcut=0.5, highcut=40.0)


def filter_ppg(data: np.ndarray, fs: float) -> np.ndarray:
    """Band-pass PPG signal(s) for pulsatile component isolation.

    Parameters
    ----------
    data : np.ndarray
        Input 1D or 2D signal data (rows represent channels for 2D input).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Filtered signal with the same dimensionality as input.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return _bandpass_filter(arr, fs, lowcut=0.4, highcut=8.0)
    if arr.ndim == 2:
        return np.vstack([_bandpass_filter(ch, fs, lowcut=0.4, highcut=8.0) for ch in arr])
    raise ValueError("PPG data must be 1D or 2D")


def sync_ecg_with_telemetry(
    ecg_features: pd.DataFrame,
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ECG summary features into the telemetry DataFrame.

    Since ECG is recorded at a much higher frequency, the extracted summary
    statistics are appended as metadata columns to the entire telemetry
    DataFrame.

    Parameters
    ----------
    ecg_features : pd.DataFrame
        Output from :func:`extract_ecg_features`.
    telemetry_df : pd.DataFrame
        Telemetry data with a ``Stage`` column.

    Returns
    -------
    pd.DataFrame
        Telemetry DataFrame augmented with ``ecg_mean_abs_uV`` and
        ``ecg_estimated_hr_bpm`` columns (mean across all ECG channels).
    """
    out = telemetry_df.copy()

    if not ecg_features.empty:
        out["ecg_mean_abs_uV"] = ecg_features["mean_abs_uV"].mean()
        out["ecg_estimated_hr_bpm"] = ecg_features["estimated_hr_bpm"].mean()

    return out


def process_patient(
    folder: Path,
    output_base: Path,
    use_ecg_checkpoint: bool = True,
    skip_ecg: bool = False,
) -> Optional[Dict]:
    """Run the full pipeline for a single patient folder.

    This is the parallelisable unit of Step 4.  It loads telemetry,
    optionally loads / extracts ECG features, synchronises the two
    data sources, saves the cleaned CSV, and returns a summary dict.

    Parameters
    ----------
    folder : Path
        Patient folder containing a ``.csv`` telemetry file and optional
        ``.bdf`` ECG files.
    output_base : Path
        Root output directory; a per-patient sub-directory is created
        automatically.
    use_ecg_checkpoint : bool
        When ``True``, load cached ECG features from disk if available,
        skipping BDF reprocessing.  Ignored when *skip_ecg* is ``True``.
    skip_ecg : bool
        When ``True``, skip all BDF discovery, ECG loading, feature
        extraction, and ECG synchronisation entirely.  The ECG helpers in
        this module remain available for standalone use.

    Returns
    -------
    Optional[Dict]
        Dict with keys ``patient_id`` (str), ``summary_row`` (dict),
        ``telemetry_df`` (DataFrame).  Returns ``None`` and logs an error
        if processing fails.
    """
    patient_id = folder.name
    logger.info("Processing patient: %s", patient_id)
    patient_out = output_base / patient_id

    try:
        # ── Load CSV ──────────────────────────────────────────────────────
        csv_path = find_csv_file(folder)
        if csv_path is None:
            raise FileNotFoundError(f"No CSV file found in {folder}")

        info_df, telemetry_df = load_telemetry(csv_path)
        logger.info("  %s — telemetry: %d rows × %d cols",
                    patient_id, *telemetry_df.shape)

        # ── Load / extract ECG features ───────────────────────────────────
        ecg_features = pd.DataFrame()
        if not skip_ecg:
            l1_path, l2_path = find_bdf_files(folder)
            ecg_features_list = []

            for bdf_label, bdf_path in [("L1", l1_path), ("L2", l2_path)]:
                if bdf_path is None:
                    logger.warning("  %s — %s BDF not found.", patient_id, bdf_label)
                    continue

                feats = None
                if use_ecg_checkpoint:
                    feats = load_ecg_checkpoint(patient_out, bdf_label)

                if feats is None:
                    raw = load_ecg(bdf_path)
                    if raw is not None:
                        feats = extract_ecg_features(raw)
                        save_ecg_checkpoint(feats, patient_out, bdf_label)

                if feats is not None:
                    feats = feats.copy()
                    feats["bdf_label"] = bdf_label
                    ecg_features_list.append(feats)

            if ecg_features_list:
                ecg_features = pd.concat(ecg_features_list, ignore_index=True)

            # ── Synchronise ECG + telemetry ───────────────────────────────
            telemetry_df = sync_ecg_with_telemetry(ecg_features, telemetry_df)

        # ── Save cleaned CSV ──────────────────────────────────────────────
        save_telemetry_csv(telemetry_df, patient_id, patient_out)

        # ── Build summary row ─────────────────────────────────────────────
        summary_row = build_patient_summary(
            patient_id,
            info_df,
            telemetry_df,
            ecg_features if not ecg_features.empty else None,
        )

        logger.info("  %s — done.", patient_id)
        return {
            "patient_id": patient_id,
            "summary_row": summary_row,
            "telemetry_df": telemetry_df,
        }

    except Exception as exc:
        logger.error("  FAILED for patient '%s': %s", patient_id, exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 3. EDA helpers
# ---------------------------------------------------------------------------


def compute_stage_summary(
    telemetry_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute mean +/- std of selected metrics grouped by trial stage.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    metrics : Optional[List[str]]
        Column names to summarise.  Defaults to all numeric columns.

    Returns
    -------
    pd.DataFrame
        MultiIndex summary with (mean, std) per stage.
    """
    if metrics is None:
        metrics = telemetry_df.select_dtypes(include=[np.number]).columns.tolist()

    available = [m for m in metrics if m in telemetry_df.columns]
    summary = (
        telemetry_df.groupby("Stage", observed=True)[available]
        .agg(["mean", "std"])
        .round(3)
    )
    return summary


def plot_metrics_by_stage(
    telemetry_df: pd.DataFrame,
    metrics: List[str],
    patient_id: str,
    output_dir: Optional[Path] = None,
) -> Figure:
    """Plot the time course of *metrics* coloured by trial stage.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    metrics : List[str]
        Column names to plot.
    patient_id : str
        Label used in the figure title and saved filename.
    output_dir : Optional[Path]
        If provided, the figure is saved as a PNG in this directory.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    available = [m for m in metrics if m in telemetry_df.columns]
    if not available:
        logger.warning("No requested metrics found for patient '%s'.", patient_id)
        fig, ax = plt.subplots()
        ax.set_title(f"No data - {patient_id}")
        return fig

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    stage_palette = dict(zip(STAGE_ORDER, sns.color_palette("tab10", len(STAGE_ORDER))))

    for ax, metric in zip(axes, available):
        for stage, grp in telemetry_df.groupby("Stage", observed=True):
            color = stage_palette.get(str(stage), "grey")
            ax.plot(grp.index, grp[metric], label=str(stage), color=color)
        ax.set_ylabel(metric, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Observation index")
    fig.suptitle(f"Patient {patient_id} - Metrics by Stage", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"{patient_id}_metrics_by_stage.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("Saved plot: %s", fig_path)

    return fig


def plot_batch_aggregates(
    summary_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> List[Figure]:
    """Generate batch-level aggregate visualisations.

    Creates:
    1. Box plot of ``VO2`` (or first available VO2-like column) grouped by
       ``Gender``.
    2. Scatter plot of ``BMI`` vs peak VO2.
    3. Scatter plot of peak VO2 vs patient ``Age``.
    4. Scatter plot of peak VO2 vs peak ``Power`` (work rate).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Master summary DataFrame with one row per patient and columns
        including ``Gender``, ``BMI``, ``Age``, ``Power``, and any
        VO2-related metric.
    output_dir : Optional[Path]
        If provided, figures are saved to this directory.

    Returns
    -------
    List[Figure]
        List of generated Matplotlib figures.
    """
    figures: List[Figure] = []

    vo2_col = _find_column(summary_df, ["VO2", "vo2", "VO2max", "vo2_max", "peak_vo2_ml_min"])
    gender_col = _find_column(summary_df, ["Gender", "gender", "Geslacht"])
    bmi_col = _find_column(summary_df, ["BMI", "bmi"])
    age_col = _find_column(summary_df, ["Age", "age", "Leeftijd"])
    power_col = _find_column(summary_df, ["Power", "power", "Watt", "watt"])

    if vo2_col and gender_col:
        fig, ax = plt.subplots(figsize=(7, 5))
        order = sorted(summary_df[gender_col].dropna().unique())
        sns.boxplot(data=summary_df, x=gender_col, y=vo2_col, order=order, ax=ax)
        ax.set_title(f"Distribution of {vo2_col} by {gender_col}")
        ax.set_xlabel(gender_col)
        ax.set_ylabel(vo2_col)
        fig.tight_layout()
        _save_figure(fig, output_dir, "batch_vo2_by_gender.png")
        figures.append(fig)

    if bmi_col and vo2_col:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(summary_df[bmi_col], summary_df[vo2_col], alpha=0.7, edgecolors="k")
        ax.set_xlabel(bmi_col)
        ax.set_ylabel(vo2_col)
        ax.set_title(f"{bmi_col} vs Peak {vo2_col}")
        fig.tight_layout()
        _save_figure(fig, output_dir, "batch_bmi_vs_vo2.png")
        figures.append(fig)

    if age_col and vo2_col:
        sub = summary_df[[age_col, vo2_col]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(sub[age_col], sub[vo2_col], alpha=0.7, edgecolors="k")
            ax.set_xlabel(f"{age_col} (years)")
            ax.set_ylabel(f"Peak {vo2_col}")
            ax.set_title(f"Peak {vo2_col} vs {age_col}")
            fig.tight_layout()
            _save_figure(fig, output_dir, "batch_vo2_vs_age.png")
            figures.append(fig)

    if power_col and vo2_col:
        sub = summary_df[[power_col, vo2_col]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(sub[power_col], sub[vo2_col], alpha=0.7, edgecolors="k")
            ax.set_xlabel(f"Peak {power_col} (W)")
            ax.set_ylabel(f"Peak {vo2_col}")
            ax.set_title(f"Peak {vo2_col} vs Peak {power_col}")
            fig.tight_layout()
            _save_figure(fig, output_dir, "batch_vo2_vs_power.png")
            figures.append(fig)

    return figures


def plot_ve_vco2_slope(
    telemetry_df: pd.DataFrame,
    patient_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot VE vs VCO₂ during active exercise stages with an OLS regression line.

    The slope of the VE/VCO₂ relationship (ventilatory efficiency) is a
    prognostically important CPET marker.  A slope < 30 is normal; >= 34
    indicates elevated ventilatory demand associated with heart failure or
    pulmonary hypertension.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    patient_id : str
        Label used in the figure title and saved filename.
    output_dir : Optional[Path]
        If provided, the figure is saved as a PNG in this directory.

    Returns
    -------
    Optional[Figure]
        Matplotlib figure, or ``None`` if required columns are absent.
    """
    ve_col = _find_column(telemetry_df, ["VE", "V'E", "V.E", "ve"])
    vco2_col = _find_column(telemetry_df, ["VCO2", "V'CO2", "V.CO2", "vco2"])
    if ve_col is None or vco2_col is None:
        logger.warning(
            "%s: VE or VCO2 column not found, skipping VE/VCO2 slope.", patient_id
        )
        return None

    active_stages = {"Test", "VT1", "VT2"}
    if "Stage" in telemetry_df.columns:
        mask = telemetry_df["Stage"].isin(active_stages)
        tdf = telemetry_df[mask] if mask.any() else telemetry_df
    else:
        tdf = telemetry_df

    tdf = (
        tdf[[ve_col, vco2_col]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if len(tdf) < 5:
        logger.warning("%s: insufficient data for VE/VCO2 slope.", patient_id)
        return None

    x = tdf[vco2_col].values
    y = tdf[ve_col].values
    x_mean, y_mean = x.mean(), y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom > 0 else float("nan")
    intercept = y_mean - slope * x_mean

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=15, alpha=0.6, label="Data (active stages)")
    x_fit = np.array([x.min(), x.max()])
    ax.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.5,
            label=f"OLS slope = {slope:.2f}")
    ax.axline((0, 0), slope=30, color="grey", linestyle="--", linewidth=1,
              label="Reference slope = 30")
    ax.set_xlabel(f"{vco2_col} (mL/min)")
    ax.set_ylabel(f"{ve_col} (L/min)")
    ax.set_title(f"{patient_id} - VE/VCO\u2082 slope")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{patient_id}_ve_vco2_slope.png")
    return fig


def plot_oxygen_pulse(
    telemetry_df: pd.DataFrame,
    patient_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot the oxygen pulse (VO₂/HR) trajectory coloured by trial stage.

    Oxygen pulse is a non-invasive proxy for stroke volume (SV × O₂
    extraction).  A plateau during increasing work rate may indicate
    inadequate cardiac output augmentation.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    patient_id : str
        Label used in the figure title and saved filename.
    output_dir : Optional[Path]
        If provided, the figure is saved as a PNG in this directory.

    Returns
    -------
    Optional[Figure]
        Matplotlib figure, or ``None`` if required columns are absent.
    """
    o2p_col = _find_column(telemetry_df, ["VO2/HR", "VO2/hr", "o2_pulse", "O2pulse"])
    vo2_col = _find_column(telemetry_df, ["VO2", "V'O2", "V.O2", "vo2"])
    hr_col = _find_column(telemetry_df, ["HR", "hr", "Heart Rate"])

    if o2p_col is not None:
        series = pd.to_numeric(telemetry_df[o2p_col], errors="coerce")
        label = o2p_col
    elif vo2_col is not None and hr_col is not None:
        vo2 = pd.to_numeric(telemetry_df[vo2_col], errors="coerce")
        hr = pd.to_numeric(telemetry_df[hr_col], errors="coerce")
        series = vo2 / hr.replace(0, float("nan"))
        label = "VO2/HR (computed)"
    else:
        logger.warning(
            "%s: VO2/HR or (VO2, HR) columns not found, skipping oxygen pulse.", patient_id
        )
        return None

    if series.dropna().empty:
        return None

    stage_col = _find_column(telemetry_df, ["Stage"])
    fig, ax = plt.subplots(figsize=(9, 4))
    stage_palette = dict(zip(STAGE_ORDER, sns.color_palette("tab10", len(STAGE_ORDER))))

    if stage_col and stage_col in telemetry_df.columns:
        for stage, grp in telemetry_df.groupby(stage_col, observed=True):
            idx = grp.index
            color = stage_palette.get(str(stage), "grey")
            ax.plot(idx, series.loc[idx], color=color, label=str(stage), linewidth=1.2)
    else:
        ax.plot(series.values, linewidth=1.2)

    ax.set_xlabel("Observation index")
    ax.set_ylabel(f"{label} (mL/beat)")
    ax.set_title(f"{patient_id} - Oxygen pulse trajectory")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{patient_id}_oxygen_pulse.png")
    return fig


def plot_ventilatory_efficiency(
    telemetry_df: pd.DataFrame,
    patient_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot VE/VO₂ and VE/VCO₂ ventilatory equivalent curves over time.

    Both curves are smoothed with a 5-point rolling mean.  The nadir of
    each curve is annotated.  The nadir of VE/VCO₂ is the clinically
    preferred ventilatory efficiency index (normal < 30).  VE/VO₂ and
    VE/VCO₂ cross near the first ventilatory threshold (VT1).

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame.
    patient_id : str
        Label used in the figure title and saved filename.
    output_dir : Optional[Path]
        If provided, the figure is saved as a PNG in this directory.

    Returns
    -------
    Optional[Figure]
        Matplotlib figure, or ``None`` if required columns are absent.
    """
    ve_col = _find_column(telemetry_df, ["VE", "V'E", "V.E", "ve"])
    vo2_col = _find_column(telemetry_df, ["VO2", "V'O2", "V.O2", "vo2"])
    vco2_col = _find_column(telemetry_df, ["VCO2", "V'CO2", "V.CO2", "vco2"])

    if ve_col is None or (vo2_col is None and vco2_col is None):
        logger.warning(
            "%s: VE and at least one of VO2/VCO2 required for ventilatory efficiency.",
            patient_id,
        )
        return None

    ve = pd.to_numeric(telemetry_df[ve_col], errors="coerce")
    curves: Dict[str, pd.Series] = {}
    if vo2_col is not None:
        vo2 = pd.to_numeric(telemetry_df[vo2_col], errors="coerce")
        ratio = ve / vo2.replace(0, float("nan"))
        curves["VE/VO\u2082"] = ratio.rolling(5, min_periods=1, center=True).mean()
    if vco2_col is not None:
        vco2 = pd.to_numeric(telemetry_df[vco2_col], errors="coerce")
        ratio = ve / vco2.replace(0, float("nan"))
        curves["VE/VCO\u2082"] = ratio.rolling(5, min_periods=1, center=True).mean()

    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["steelblue", "darkorange"]
    for (curve_name, curve), color in zip(curves.items(), colors):
        clean = curve.dropna()
        if clean.empty:
            continue
        positions = [clean.index.get_loc(i) for i in clean.index]
        ax.plot(positions, clean.values, label=curve_name, color=color, linewidth=1.5)
        nadir_idx = clean.idxmin()
        nadir_pos = clean.index.get_loc(nadir_idx)
        nadir_val = clean[nadir_idx]
        ax.annotate(
            f"nadir {nadir_val:.1f}",
            xy=(nadir_pos, nadir_val),
            xytext=(nadir_pos + max(1, int(len(clean) * 0.05)), nadir_val * 1.05),
            arrowprops=dict(arrowstyle="->", color=color),
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Observation index")
    ax.set_ylabel("Ventilatory equivalent")
    ax.set_title(f"{patient_id} - Ventilatory efficiency (VE/VO\u2082 and VE/VCO\u2082)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{patient_id}_ventilatory_efficiency.png")
    return fig


def compute_breathing_reserve(
    telemetry_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute breathing reserve from peak ventilation during active exercise stages.

    MVV is estimated as ``peak_VE / 0.70``, which assumes a 30 % breathing
    reserve at maximum exercise (the clinically accepted approximation when
    spirometry MVV is unavailable).  A breathing reserve < 15 % (or < 11 L/min)
    suggests a ventilatory limitation to exercise.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:

        * ``peak_VE``  — peak minute ventilation in L/min (NaN if absent)
        * ``MVV_est``  — estimated maximal voluntary ventilation in L/min
        * ``BR_L``     — breathing reserve in L/min (MVV_est - peak_VE)
        * ``BR_pct``   — breathing reserve as percentage of MVV_est
    """
    ve_col = _find_column(telemetry_df, ["VE", "V'E", "V.E", "ve"])
    nan_result: Dict[str, float] = {
        "peak_VE": float("nan"),
        "MVV_est": float("nan"),
        "BR_L": float("nan"),
        "BR_pct": float("nan"),
    }
    if ve_col is None:
        return nan_result

    active_stages = {"Test", "VT1", "VT2"}
    if "Stage" in telemetry_df.columns:
        mask = telemetry_df["Stage"].isin(active_stages)
        active_df = telemetry_df[mask] if mask.any() else telemetry_df
    else:
        active_df = telemetry_df

    ve = pd.to_numeric(active_df[ve_col], errors="coerce").dropna()
    if ve.empty:
        return nan_result

    peak_ve = float(ve.max())
    mvv_est = peak_ve / 0.70
    br_l = mvv_est - peak_ve
    br_pct = (br_l / mvv_est) * 100.0 if mvv_est > 0 else float("nan")

    return {
        "peak_VE": round(peak_ve, 2),
        "MVV_est": round(mvv_est, 2),
        "BR_L": round(br_l, 2),
        "BR_pct": round(br_pct, 1),
    }


# ---------------------------------------------------------------------------
# 4. Machine learning helpers
# ---------------------------------------------------------------------------


def prepare_features(
    telemetry_df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """Prepare a scaled feature matrix from telemetry data.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame.
    drop_cols : Optional[List[str]]
        Additional column names to exclude from features.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, StandardScaler]
        ``(X_scaled, feature_df, scaler)`` where ``feature_df`` is the
        imputed (but unscaled) numeric DataFrame and ``scaler`` is the fitted
        :class:`~sklearn.preprocessing.StandardScaler`.
    """
    exclude = set(["Stage"] + (drop_cols or []))
    num_df = telemetry_df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in exclude if c in telemetry_df.columns],
        errors="ignore",
    )

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(num_df)
    feature_df = pd.DataFrame(X_imputed, columns=num_df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, feature_df, scaler


def run_pca(
    X_scaled: np.ndarray,
    n_components: int = 10,
) -> Tuple[PCA, np.ndarray]:
    """Fit PCA on scaled data.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix of shape ``(n_samples, n_features)``.
    n_components : int
        Number of principal components to retain (capped at
        ``min(n_samples, n_features)``).

    Returns
    -------
    Tuple[PCA, np.ndarray]
        ``(pca_model, X_pca)`` where ``X_pca`` has shape
        ``(n_samples, n_components)``.
    """
    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def plot_scree(
    pca: PCA,
    patient_id: str = "",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Generate a Scree plot for a fitted PCA model.

    Parameters
    ----------
    pca : PCA
        Fitted :class:`~sklearn.decomposition.PCA` instance.
    patient_id : str
        Label used in the title and saved filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    ev = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)
    components = np.arange(1, len(ev) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(components, ev * 100, color="steelblue", alpha=0.8, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title(f"Scree Plot - {patient_id}" if patient_id else "Scree Plot")

    ax2 = ax1.twinx()
    ax2.plot(components, cum_ev * 100, color="tomato", marker="o", label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.axhline(y=90, color="tomato", linestyle="--", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    _save_figure(fig, output_dir, f"{patient_id}_scree.png" if patient_id else "scree.png")
    return fig


def plot_pca_scatter(
    X_pca: np.ndarray,
    labels: pd.Series,
    label_name: str = "Stage",
    patient_id: str = "",
    output_dir: Optional[Path] = None,
    three_d: bool = False,
) -> Figure:
    """2-D or 3-D scatter plot of the first two/three principal components.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-transformed data of shape ``(n_samples, >= 2)``.
    labels : pd.Series
        Categorical labels for colouring (e.g., stage or gender).
    label_name : str
        Display name for the colour legend.
    patient_id : str
        Label used in the title and filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.
    three_d : bool
        If ``True`` and at least 3 PCs are available, produces a 3-D plot.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    unique_labels = labels.unique()
    palette = dict(
        zip(unique_labels, sns.color_palette("tab10", len(unique_labels)))
    )
    colors = labels.map(palette).values

    if three_d and X_pca.shape[1] >= 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for lbl, color in palette.items():
            mask = labels == lbl
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                X_pca[mask, 2],
                label=str(lbl),
                color=color,
                alpha=0.7,
                s=20,
            )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        title_suffix = "3D"
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl, color in palette.items():
            mask = labels == lbl
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=str(lbl),
                color=color,
                alpha=0.7,
                s=20,
            )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        title_suffix = "2D"
        ax.grid(True, alpha=0.3)

    title = (
        f"PCA {title_suffix} - {patient_id} ({label_name})"
        if patient_id
        else f"PCA {title_suffix} ({label_name})"
    )
    if three_d and X_pca.shape[1] >= 3:
        fig.suptitle(title)
    else:
        ax.set_title(title)  # type: ignore[union-attr]
    plt.legend(title=label_name, fontsize=8)
    fig.tight_layout()

    fname = (
        f"{patient_id}_pca_{title_suffix.lower()}_{label_name.lower()}.png"
        if patient_id
        else f"pca_{title_suffix.lower()}_{label_name.lower()}.png"
    )
    _save_figure(fig, output_dir, fname)
    return fig


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
) -> Tuple[KMeans, np.ndarray]:
    """Fit K-Means clustering.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple[KMeans, np.ndarray]
        ``(model, cluster_labels)``
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def run_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Tuple[DBSCAN, np.ndarray]:
    """Fit DBSCAN clustering.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    eps : float
        Maximum distance between two samples in the same neighbourhood.
    min_samples : int
        Minimum number of samples in a neighbourhood.

    Returns
    -------
    Tuple[DBSCAN, np.ndarray]
        ``(model, cluster_labels)``
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return db, labels


def plot_cluster_scatter(
    feature_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    x_col: str,
    y_col: str,
    patient_id: str = "",
    algorithm_name: str = "Cluster",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Scatter plot of two physiological metrics coloured by cluster assignment.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Un-scaled feature DataFrame (for axis labels and values).
    cluster_labels : np.ndarray
        Integer cluster assignments from a clustering model.
    x_col : str
        Column to use as the X axis.
    y_col : str
        Column to use as the Y axis.
    patient_id : str
        Label used in the title.
    algorithm_name : str
        Name of the clustering algorithm (for the title).
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if x_col not in feature_df.columns or y_col not in feature_df.columns:
        ax.set_title("Requested columns not available")
        return fig

    unique_clusters = np.unique(cluster_labels)
    palette = dict(
        zip(unique_clusters, sns.color_palette("tab10", len(unique_clusters)))
    )

    for clust in unique_clusters:
        mask = cluster_labels == clust
        lbl = f"Cluster {clust}" if clust >= 0 else "Noise"
        color = palette[clust]
        ax.scatter(
            feature_df.loc[mask, x_col],
            feature_df.loc[mask, y_col],
            label=lbl,
            color=color,
            alpha=0.7,
            s=25,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(
        f"{algorithm_name} Clusters - {patient_id}" if patient_id else f"{algorithm_name} Clusters"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = (
        f"{patient_id}_{algorithm_name.lower()}_clusters.png"
        if patient_id
        else f"{algorithm_name.lower()}_clusters.png"
    )
    _save_figure(fig, output_dir, fname)
    return fig


def elbow_plot(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    patient_id: str = "",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Generate an elbow plot to aid K-Means cluster selection.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of K values to evaluate.
    patient_id : str
        Label used in the title and filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    inertias = []
    for k in tqdm(k_range, desc="Elbow (K-Means)"):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", color="steelblue")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title(f"Elbow Plot - {patient_id}" if patient_id else "Elbow Plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"{patient_id}_elbow.png" if patient_id else "elbow.png"
    _save_figure(fig, output_dir, fname)
    return fig


# ---------------------------------------------------------------------------
# 5. Export helpers
# ---------------------------------------------------------------------------


def save_telemetry_csv(
    telemetry_df: pd.DataFrame,
    patient_id: str,
    output_dir: Path,
) -> Path:
    """Save cleaned telemetry data as a CSV file.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame.
    patient_id : str
        Used to construct the output filename.
    output_dir : Path
        Target directory (created if it does not exist).

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{patient_id}_telemetry.csv"
    telemetry_df.to_csv(out_path, index=False)
    logger.info("Saved telemetry CSV: %s", out_path)
    return out_path


def build_patient_summary(
    patient_id: str,
    info_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    ecg_features: Optional[pd.DataFrame] = None,
) -> Dict:
    """Compile a single-row summary dict for one patient.

    Parameters
    ----------
    patient_id : str
        Unique patient / folder identifier.
    info_df : pd.DataFrame
        Patient metadata DataFrame.
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame.
    ecg_features : Optional[pd.DataFrame]
        ECG summary features (optional).

    Returns
    -------
    Dict
        Flat dictionary suitable for appending to a master summary DataFrame.
    """
    row: Dict = {"patient_id": patient_id}

    # --- Patient info (first row) ---
    if not info_df.empty:
        meta_row = info_df.iloc[0]
        for col in ["Age", "BMI", "Weight", "Gender"]:
            match = _find_column_in_series(meta_row, [col])
            if match:
                row[col.lower()] = meta_row[match]

    # --- Peak telemetry values per numeric column ---
    numeric_cols = telemetry_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        row[f"peak_{col}"] = telemetry_df[col].max()
        row[f"mean_{col}"] = telemetry_df[col].mean()

    # --- ECG summary ---
    if ecg_features is not None and not ecg_features.empty:
        row["ecg_mean_abs_uV"] = ecg_features["mean_abs_uV"].mean()
        row["ecg_estimated_hr_bpm"] = ecg_features["estimated_hr_bpm"].mean()

    return row


# ---------------------------------------------------------------------------
# COPD risk scoring
# ---------------------------------------------------------------------------

#: Clinical thresholds derived from ATS/ERS CPET guidelines and COPD literature.
#: Sources: ATS/ACCP Statement on CPET (Am J Respir Crit Care Med 2003);
#:          Puente-Maestu et al. ERJ 2016; Wasserman et al. "Principles of
#:          Exercise Testing and Interpretation" 5th ed.
_COPD_THRESHOLDS: Dict[str, float] = {
    "peak_vo2_per_kg_low": 20.0,   # mL/min/kg  — below = elevated risk
    "ve_vco2_slope_high": 34.0,    # above = elevated ventilatory inefficiency
    "spo2_nadir_low": 95.0,        # % — below = exercise-induced desaturation
    "spo2_drop_high": 4.0,         # % — drop ≥ 4 pp = clinically significant
    "bf_peak_high": 40.0,          # breaths/min — above = severe hyperventilation
    "rer_peak_low": 1.0,           # below at exhaustion = effort-limited test
    "o2_pulse_peak_low": 10.0,     # mL/beat — below = reduced cardiac output
}


def parse_patient_meta(raw_df: pd.DataFrame) -> Dict:
    """Extract patient metadata from the 5-row CSV header block.

    The first five rows of a WESENSE CPET export contain key–value pairs
    encoded across alternating columns, for example::

        Identificatie: | WESENSETEST04 | Naam: | ... | Voornaam: | ...
        Bezoekdatum:   | 09.01.2026    | Geboortedatum: | 1-1-1997 | Leeftijd: | 29 Jaar
        Geslacht:      | vrouw         | Gewicht: | 68.0 kg | BMI: | 24 kg/m²
        Gebruiker:     | ...

    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame as returned by ``pd.read_csv(path, header=0)`` — i.e. the
        first data row becomes the header and subsequent rows are data.

    Returns
    -------
    Dict
        Keys: ``patient_id``, ``visit_date``, ``birth_date``, ``age_years``,
        ``sex``, ``weight_kg``, ``bmi``.  Missing values are ``None``.
    """
    meta: Dict = {
        "patient_id": None,
        "visit_date": None,
        "birth_date": None,
        "age_years": None,
        "sex": None,
        "weight_kg": None,
        "bmi": None,
    }

    # The header itself (row 0 of the file) becomes column names; rows 0-2 of
    # raw_df correspond to file rows 1-3 (0-indexed).
    def _cell(row_idx: int, col_idx: int) -> str:
        try:
            return str(raw_df.iloc[row_idx, col_idx]).strip()
        except (IndexError, KeyError):
            return ""

    # Row 0 (file row 1): Identificatie / Naam / Voornaam
    meta["patient_id"] = _cell(0, 1) or None

    # Row 1 (file row 2): Bezoekdatum / Geboortedatum / Leeftijd
    meta["visit_date"] = _cell(1, 1) or None
    meta["birth_date"] = _cell(1, 3) or None
    age_raw = _cell(1, 5)
    try:
        meta["age_years"] = float(age_raw.split()[0])
    except (ValueError, IndexError):
        pass

    # Row 2 (file row 3): Geslacht / Gewicht / BMI
    meta["sex"] = _cell(2, 1) or None
    weight_raw = _cell(2, 3)
    try:
        meta["weight_kg"] = float(weight_raw.split()[0])
    except (ValueError, IndexError):
        pass
    bmi_raw = _cell(2, 5)
    try:
        meta["bmi"] = float(bmi_raw.split()[0])
    except (ValueError, IndexError):
        pass

    return meta


def extract_copd_features(
    telemetry_df: pd.DataFrame,
    weight_kg: Optional[float] = None,
) -> pd.Series:
    """Compute CPET-based markers associated with COPD severity.

    All markers are derived from the cleaned telemetry DataFrame that
    ``load_telemetry`` returns.  The ``Stage`` column is used to restrict
    certain calculations to the relevant exercise phase.

    Parameters
    ----------
    telemetry_df : pd.DataFrame
        Cleaned telemetry DataFrame with a ``Stage`` column.
    weight_kg : Optional[float]
        Patient body weight in kg.  Required to compute peak VO₂/kg; if
        ``None`` the marker is ``NaN``.

    Returns
    -------
    pd.Series
        Named Series with the following keys:

        * ``peak_vo2_ml_min``      — peak oxygen uptake (mL/min)
        * ``peak_vo2_per_kg``      — peak VO₂ normalised by body weight
        * ``ve_vco2_slope``        — slope of V′E vs V′CO₂ (linear regression)
        * ``spo2_nadir``           — minimum SpO₂ during the test (%)
        * ``spo2_drop``            — baseline SpO₂ − nadir (percentage points)
        * ``bf_peak``              — peak breathing frequency (breaths/min)
        * ``rer_peak``             — peak respiratory exchange ratio
        * ``o2_pulse_peak``        — peak oxygen pulse (mL/beat)
        * ``vt1_present``          — 1.0 if VT1 stage present, else 0.0
    """
    df = telemetry_df.copy()

    def _col(candidates: List[str]) -> Optional[str]:
        return _find_column(df, candidates)

    vo2_col = _col(["V'O2", "VO2", "V.O2", "vo2"])
    vco2_col = _col(["V'CO2", "VCO2", "V.CO2", "vco2"])
    ve_col = _col(["V'E", "VE", "V.E", "ve"])
    spo2_col = _col(["SpO2", "SPO2", "spo2"])
    bf_col = _col(["BF", "bf", "RR"])
    rer_col = _col(["RER", "rer"])
    o2_pulse_col = _col(["VO2/HR", "VO2/hr", "o2_pulse", "O2pulse"])

    # Exercise phases to use for "peak" values (exclude warm-up and recovery)
    active_stages = {"Test", "VT1", "VT2"}
    if "Stage" in df.columns:
        active_mask = df["Stage"].isin(active_stages)
        active_df = df[active_mask] if active_mask.any() else df
    else:
        active_df = df

    # ── Peak VO₂ ──────────────────────────────────────────────────────────
    peak_vo2 = float(
        pd.to_numeric(active_df[vo2_col], errors="coerce").max()
    ) if vo2_col else float("nan")
    if not np.isnan(peak_vo2) and weight_kg and weight_kg > 0:
        peak_vo2_per_kg = peak_vo2 / weight_kg
    else:
        peak_vo2_per_kg = float("nan")

    # ── VE/VCO₂ slope (linear regression over active stages) ──────────────
    ve_vco2_slope = float("nan")
    if ve_col and vco2_col:
        subset = active_df[[ve_col, vco2_col]].copy()
        subset[ve_col] = pd.to_numeric(subset[ve_col], errors="coerce")
        subset[vco2_col] = pd.to_numeric(subset[vco2_col], errors="coerce")
        subset = subset.dropna()
        if len(subset) >= 5:
            x = subset[vco2_col].values.astype(float)
            y = subset[ve_col].values.astype(float)
            # Least-squares slope via normal equations (avoid sklearn dependency)
            x_mean, y_mean = x.mean(), y.mean()
            denom = float(np.sum((x - x_mean) ** 2))
            if denom > 0:
                ve_vco2_slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    # ── SpO₂ nadir and drop ───────────────────────────────────────────────
    spo2_nadir = float("nan")
    spo2_drop = float("nan")
    if spo2_col:
        spo2_series = pd.to_numeric(df[spo2_col], errors="coerce")
        if not spo2_series.dropna().empty:
            # Baseline: median of first 10 chronological rows (resting / warm-up)
            baseline_spo2 = float(spo2_series.head(10).dropna().median())
            spo2_nadir = float(
                pd.to_numeric(active_df[spo2_col], errors="coerce").min()
            )
            spo2_drop = max(0.0, baseline_spo2 - spo2_nadir)

    # ── Peak breathing frequency ───────────────────────────────────────────
    bf_peak = (
        float(pd.to_numeric(active_df[bf_col], errors="coerce").max())
        if bf_col else float("nan")
    )

    # ── Peak RER ──────────────────────────────────────────────────────────
    rer_peak = (
        float(pd.to_numeric(active_df[rer_col], errors="coerce").max())
        if rer_col else float("nan")
    )

    # ── Peak oxygen pulse ─────────────────────────────────────────────────
    o2_pulse_peak = (
        float(pd.to_numeric(active_df[o2_pulse_col], errors="coerce").max())
        if o2_pulse_col else float("nan")
    )

    # ── VT1 presence ──────────────────────────────────────────────────────
    vt1_present = (
        1.0
        if "Stage" in df.columns and (df["Stage"] == "VT1").any()
        else 0.0
    )

    return pd.Series(
        {
            "peak_vo2_ml_min": peak_vo2,
            "peak_vo2_per_kg": peak_vo2_per_kg,
            "ve_vco2_slope": ve_vco2_slope,
            "spo2_nadir": spo2_nadir,
            "spo2_drop": spo2_drop,
            "bf_peak": bf_peak,
            "rer_peak": rer_peak,
            "o2_pulse_peak": o2_pulse_peak,
            "vt1_present": vt1_present,
        }
    )


def score_copd_risk(features: pd.Series) -> pd.Series:
    """Apply clinical thresholds to CPET features and compute a composite risk score.

    Each marker is compared against the thresholds in :data:`_COPD_THRESHOLDS`.
    A flag of ``1`` indicates an abnormal value associated with elevated risk.
    The composite score is the sum of all flags, mapped to an ordinal label.

    Parameters
    ----------
    features : pd.Series
        As returned by :func:`extract_copd_features`.

    Returns
    -------
    pd.Series
        Original feature values plus flag columns (``flag_*``) and:

        * ``n_flags``       — total number of raised flags (int)
        * ``risk_score``    — ``"Low"``, ``"Moderate"``, or ``"High"``
    """
    result = features.copy()
    t = _COPD_THRESHOLDS

    result["flag_low_peak_vo2"] = int(
        not np.isnan(features["peak_vo2_per_kg"])
        and features["peak_vo2_per_kg"] < t["peak_vo2_per_kg_low"]
    )
    result["flag_high_ve_vco2"] = int(
        not np.isnan(features["ve_vco2_slope"])
        and features["ve_vco2_slope"] > t["ve_vco2_slope_high"]
    )
    result["flag_low_spo2"] = int(
        not np.isnan(features["spo2_nadir"])
        and features["spo2_nadir"] < t["spo2_nadir_low"]
    )
    result["flag_spo2_drop"] = int(
        not np.isnan(features["spo2_drop"])
        and features["spo2_drop"] >= t["spo2_drop_high"]
    )
    result["flag_high_bf"] = int(
        not np.isnan(features["bf_peak"])
        and features["bf_peak"] > t["bf_peak_high"]
    )
    result["flag_low_rer"] = int(
        not np.isnan(features["rer_peak"])
        and features["rer_peak"] < t["rer_peak_low"]
    )
    result["flag_low_o2_pulse"] = int(
        not np.isnan(features["o2_pulse_peak"])
        and features["o2_pulse_peak"] < t["o2_pulse_peak_low"]
    )

    flag_cols = [c for c in result.index if c.startswith("flag_")]
    n_flags = int(sum(result[c] for c in flag_cols))
    result["n_flags"] = n_flags

    if n_flags == 0:
        risk = "Low"
    elif n_flags <= 2:
        risk = "Moderate"
    else:
        risk = "High"
    result["risk_score"] = risk

    return result


def plot_copd_radar(
    features: pd.Series,
    patient_id: str = "",
    output_dir: Optional[Path] = None,
) -> Figure:
    """Spider / radar chart of normalised COPD marker values.

    Each marker is scaled to [0, 1] relative to its clinical threshold so
    that values beyond the threshold appear in the outer (red) zone.

    Parameters
    ----------
    features : pd.Series
        As returned by :func:`extract_copd_features` (raw values, not flags).
    patient_id : str
        Used in the figure title and output filename.
    output_dir : Optional[Path]
        If provided, the figure is saved here.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    t = _COPD_THRESHOLDS

    # Each tuple: (label, value, threshold, higher_is_worse)
    markers = [
        ("Peak VO₂/kg\n(mL/min/kg)", features.get("peak_vo2_per_kg", float("nan")),
         t["peak_vo2_per_kg_low"], False),
        ("VE/VCO₂\nslope", features.get("ve_vco2_slope", float("nan")),
         t["ve_vco2_slope_high"], True),
        ("SpO₂ nadir\n(%)", features.get("spo2_nadir", float("nan")),
         t["spo2_nadir_low"], False),
        ("SpO₂ drop\n(pp)", features.get("spo2_drop", float("nan")),
         t["spo2_drop_high"], True),
        ("Peak BF\n(br/min)", features.get("bf_peak", float("nan")),
         t["bf_peak_high"], True),
        ("Peak RER", features.get("rer_peak", float("nan")),
         t["rer_peak_low"], False),
        ("O₂ pulse\n(mL/beat)", features.get("o2_pulse_peak", float("nan")),
         t["o2_pulse_peak_low"], False),
    ]

    labels = [m[0] for m in markers]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalise: threshold maps to 0.5; clip to [0, 1]
    norm_values: List[float] = []
    for _, val, thresh, higher_worse in markers:
        if np.isnan(val) or thresh == 0:
            norm_values.append(0.5)
            continue
        if higher_worse:
            # value/threshold * 0.5  →  1.0 means 2× threshold
            norm = np.clip(val / thresh * 0.5, 0.0, 1.0)
        else:
            # thresh/value * 0.5  →  1.0 means value = 0.5 × threshold
            norm = np.clip(thresh / val * 0.5, 0.0, 1.0) if val > 0 else 1.0
        norm_values.append(float(norm))
    norm_values += norm_values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "threshold", "", ""], fontsize=7)

    # Reference circle at threshold (0.5)
    ax.plot([a for a in angles], [0.5] * len(angles), color="grey",
            linestyle="--", linewidth=0.8, alpha=0.6)

    ax.plot(angles, norm_values, color="steelblue", linewidth=2)
    ax.fill(angles, norm_values, color="steelblue", alpha=0.25)

    title = f"COPD Marker Profile — {patient_id}" if patient_id else "COPD Marker Profile"
    ax.set_title(title, pad=20, fontsize=10)
    fig.tight_layout()

    fname = f"{patient_id}_copd_radar.png" if patient_id else "copd_radar.png"
    _save_figure(fig, output_dir, fname)
    return fig


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column from *candidates* that exists in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    candidates : List[str]
        Candidate column names in priority order.

    Returns
    -------
    Optional[str]
        Matched column name, or ``None``.
    """
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive fallback.
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _find_column_in_series(series: pd.Series, candidates: List[str]) -> Optional[str]:
    """Return the first matching index label from *candidates* in *series*.

    Parameters
    ----------
    series : pd.Series
        Series whose index to search.
    candidates : List[str]
        Candidate label names in priority order.

    Returns
    -------
    Optional[str]
        Matched label, or ``None``.
    """
    for c in candidates:
        if c in series.index:
            return c
    lower_map = {str(idx).lower(): idx for idx in series.index}
    for c in candidates:
        if c.lower() in lower_map:
            return str(lower_map[c.lower()])
    return None


def series_to_table(series: pd.Series) -> pd.DataFrame:
    """Convert a Series into a 2-column DataFrame for display/export.

    Parameters
    ----------
    series : pd.Series
        Source Series whose index contains metric names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``metric`` and ``value``.
    """
    return series.rename("value").reset_index(names="metric")


def _save_figure(fig: Figure, output_dir: Optional[Path], filename: str) -> None:
    """Save *fig* to *output_dir/filename* if *output_dir* is not ``None``.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    output_dir : Optional[Path]
        Target directory.
    filename : str
        Output filename including extension.
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", out_path)


def show_and_save_figure(fig: Figure, output_dir: Optional[Path], filename: str) -> None:
    """Save a figure to PNG and show it in interactive contexts.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to render.
    output_dir : Optional[Path]
        Directory where the PNG file should be written.
    filename : str
        Output PNG filename.
    """
    _save_figure(fig, output_dir, filename)
    plt.show()


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format.

    Parameters
    ----------
    log_file : Optional[str]
        If provided, log messages are also written to this file.
    level : int
        Logging level (default: ``logging.INFO``).
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def convert_xls_to_csv(xls_path: Path, csv_path: Path) -> None:
    """Convert a tab-separated ISO-8859-1 ``.xls`` file to UTF-8 CSV.

    Parameters
    ----------
    xls_path : Path
        Input telemetry export path.
    csv_path : Path
        Output CSV path.
    """
    with xls_path.open(encoding="iso-8859-1", newline="") as infile:
        reader = csv.reader(infile, delimiter="\t")
        with csv_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow(row)
