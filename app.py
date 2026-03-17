"""Streamlit GUI for CPET and ECG exploration."""

from pathlib import Path

import streamlit as st

import functions as fn


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="DACIL-WESENSE GUI", layout="wide")
    st.title("DACIL-WESENSE Quantitative Analysis")

    fn.setup_logging()

    st.sidebar.header("Configuration")
    data_root = st.sidebar.text_input("Data root", "data")
    output_root = st.sidebar.text_input("Output root", "output")

    folders = fn.discover_patient_folders(data_root)
    if not folders:
        st.warning(f"No patient folders found under: {data_root}")
        return

    patient_map = {folder.name: folder for folder in folders}
    patient_name = st.sidebar.selectbox("Patient", options=list(patient_map.keys()))
    patient_folder = patient_map[patient_name]
    output_dir = Path(output_root)

    st.subheader(f"Patient: {patient_name}")
    csv_path = fn.find_csv_file(patient_folder)
    if csv_path is None:
        st.error("No telemetry CSV file found in the selected patient folder.")
        return

    info_df, telemetry_df = fn.load_telemetry(csv_path)

    with st.expander("Raw metadata"):
        st.dataframe(info_df, use_container_width=True)

    meta = fn.parse_patient_meta(info_df)
    st.write(
        {
            "patient_id": meta.get("patient_id"),
            "age_years": meta.get("age_years"),
            "sex": meta.get("sex"),
            "weight_kg": meta.get("weight_kg"),
            "bmi": meta.get("bmi"),
        }
    )

    st.subheader("Telemetry preview")
    st.dataframe(telemetry_df.head(200), use_container_width=True)

    default_metrics = ["HR", "RER", "SpO2", "V'O2", "V'CO2", "V'E", "Belasting", "BF"]
    available_defaults = [m for m in default_metrics if m in telemetry_df.columns]
    metrics = st.multiselect(
        "Metrics to plot by stage",
        options=list(telemetry_df.columns),
        default=available_defaults,
    )
    if metrics:
        fig = fn.plot_metrics_by_stage(
            telemetry_df=telemetry_df,
            metrics=metrics,
            patient_id=patient_name,
            output_dir=output_dir / "streamlit",
        )
        st.pyplot(fig, use_container_width=True)

    st.subheader("COPD risk")
    features = fn.extract_copd_features(telemetry_df, weight_kg=meta.get("weight_kg"))
    score = fn.score_copd_risk(features)
    st.dataframe(fn.series_to_table(features), use_container_width=True)
    st.dataframe(fn.series_to_table(score), use_container_width=True)

    radar = fn.plot_copd_radar(features=features, patient_id=patient_name)
    st.pyplot(radar, use_container_width=False)

    st.subheader("ECG files")
    l1_path, l2_path = fn.find_bdf_files(patient_folder)
    ecg_files = [("L1", l1_path), ("L2", l2_path)]
    for label, bdf_path in ecg_files:
        if bdf_path is None:
            st.info(f"{label}: not found")
            continue
        st.write(f"{label}: {bdf_path.name}")
        raw = fn.load_ecg(bdf_path)
        if raw is None:
            st.warning(f"{label}: could not load file")
            continue
        ecg_summary = fn.extract_ecg_features(raw)
        st.dataframe(ecg_summary, use_container_width=True)


if __name__ == "__main__":
    main()
