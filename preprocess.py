import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

CATEGORY_MAP = {
    'google nest mini speaker': 'Audio',
    'amazon echo dot 1': 'Audio',
    'amazon echo dot 2': 'Audio',
    'amazon echo show': 'Audio',
    'amazon echo studio': 'Audio',
    'harman kardon (ampak technology)': 'Audio',

    'wyze camera': 'Camera',
    'arlo q indoor camera': 'Camera',
    'nest indoor camera': 'Camera',
    'yi indoor 2 camera': 'Camera',
    'home eye camera': 'Camera',

    'smartthings hub': 'Hub',
    'aeotec smart home hub': 'Hub',

    'philips hue bridge': 'Hub_Lighting',
}

FULL_FEATURES = [
    'inter_arrival_time', 'time_since_previously_displayed_frame',
    'port_class_dst', 'l4_tcp', 'l4_udp', 'ttl', 'eth_size',
    'tcp_window_size', 'payload_entropy', 'handshake_version',
    'handshake_cipher_suites_length', 'handshake_extensions_length',
    'handshake_sig_hash_alg_len', 'http_response_code',
    'dns_query_type', 'dns_len_qry', 'dns_interval', 'dns_len_ans',
    'payload_length', 'highest_layer', 'http_content_len',
    'icmp_type', 'icmp_data_size', 'jitter',
    'stream_1_count', 'stream_1_mean', 'stream_1_var',
    'src_ip_1_count', 'src_ip_1_mean', 'src_ip_1_var',
    'channel_1_count', 'channel_1_mean', 'channel_1_var',
    'stream_jitter_1_sum', 'stream_jitter_1_mean', 'stream_jitter_1_var',
    'stream_5_count', 'stream_5_mean', 'stream_5_var',
    'src_ip_5_count', 'src_ip_5_mean', 'src_ip_5_var',
    'channel_5_count', 'channel_5_mean', 'channel_5_var',
    'stream_jitter_5_sum', 'stream_jitter_5_mean', 'stream_jitter_5_var',
    'stream_10_count', 'stream_10_mean', 'stream_10_var',
    'src_ip_10_count', 'src_ip_10_mean', 'src_ip_10_var',
    'channel_10_count', 'channel_10_mean', 'channel_10_var',
    'stream_jitter_10_sum', 'stream_jitter_10_mean', 'stream_jitter_10_var',
    'stream_30_count', 'stream_30_mean', 'stream_30_var',
    'src_ip_30_count', 'src_ip_30_mean', 'src_ip_30_var',
    'channel_30_count', 'channel_30_mean', 'channel_30_var',
    'stream_jitter_30_sum', 'stream_jitter_30_mean', 'stream_jitter_30_var',
    'stream_60_count', 'stream_60_mean', 'stream_60_var',
    'src_ip_60_count', 'src_ip_60_mean', 'src_ip_60_var',
    'channel_60_count', 'channel_60_mean', 'channel_60_var',
    'stream_jitter_60_sum', 'stream_jitter_60_mean', 'stream_jitter_60_var',
    'ntp_interval', 'most_freq_spot', 'min_et', 'q1', 'min_e',
    'var_e', 'q1_e', 'sum_p', 'min_p', 'max_p', 'med_p',
    'average_p', 'var_p', 'q3_p', 'q1_p', 'iqr_p',
    'l3_ip_dst_count', 'eth_src_oui', 'eth_dst_oui',
]

ID_COLUMNS = {
    'src_mac', 'dst_mac', 'src_ip', 'dst_ip',
    'src_port', 'dst_port', 'stream',
    'eth_src_oui', 'eth_dst_oui',
    'tls_server', 'http_host', 'http_uri',
    'dns_server', 'user_agent', 'http_content_type',
    'handshake_ciphersuites',
}

NO_ID_FEATURES = [f for f in FULL_FEATURES if f not in ID_COLUMNS]

BEHAVIORAL_FEATURES = [
    'inter_arrival_time', 'time_since_previously_displayed_frame',
    'eth_size', 'payload_length', 'payload_entropy', 'jitter',
    'tcp_window_size', 'ttl',
    'stream_1_count', 'stream_1_mean', 'stream_1_var',
    'stream_5_count', 'stream_5_mean', 'stream_5_var',
    'stream_10_count', 'stream_10_mean', 'stream_10_var',
    'stream_30_count', 'stream_30_mean', 'stream_30_var',
    'stream_60_count', 'stream_60_mean', 'stream_60_var',
    'stream_jitter_1_mean', 'stream_jitter_5_mean',
    'stream_jitter_10_mean', 'stream_jitter_30_mean',
    'stream_jitter_60_mean',
    'ntp_interval', 'dns_interval',
    'sum_p', 'min_p', 'max_p', 'med_p', 'average_p', 'var_p', 'iqr_p',
]

FEATURE_SETS = {
    'full': FULL_FEATURES,
    'no_id': NO_ID_FEATURES,
    'behavioral': BEHAVIORAL_FEATURES,
}


def load_directory(directory):

    pattern = os.path.join(directory, 'BenignTraffic*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No BenignTraffic*.csv files found in {directory}"
        )
    frames = []
    for f in files:
        print(f"Loading {os.path.basename(f)}...")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")
    return df


def assign_labels(df):

    df = df.copy()
    df['device_mac'] = df['device_mac'].str.lower().str.strip()
    df['device_name'] = df['device_mac']
    df['category'] = df['device_mac'].map(CATEGORY_MAP)  # Audio, Camera, ... blah, etc.
    before = len(df)
    df = df[df['device_name'].notna()].copy()
    df = df[df['category'].notna()].copy()
    print(f"Kept {len(df):,} rows that have known labels"
          f"(dropped {before - len(df):,} unknown)")
    return df


def encode_features(df, feature_cols, encoders=None, fit=True):

    df = df.copy()
    present = [c for c in feature_cols if c in df.columns]
    missmissmiss = set(feature_cols) - set(present)
    if missmissmiss:
        print(f"Note: {len(missmissmiss)} features not in data, SKIPPING.")

    df = df[present].replace(['-', '', 'None', 'N/A'], np.nan).fillna(0)
    categorical = df.select_dtypes(include=['object']).columns.tolist()

    if encoders is None:
        encoders = {}

    for col in categorical:
        df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                df[col] = 0
            else:
                df[col] = df[col].apply(
                    lambda v: le.transform([v])[0]
                    if v in le.classes_ else 0
                )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.values, encoders, present


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        # for TA
        print("usage: python preprocess.py <benign_dir> [benign_dir2 ...]")
        print("example (2024 only): python preprocess.py CIC-IoT-2024/Benign_Final/")
        print("example (both datasets): python preprocess.py CIC-IoT-2023/Benign_Final/ "
              "CIC-IoT-2024/Benign_Final/")
        sys.exit(1)

    frames = []
    for d in sys.argv[1:]:
        print(f"\nLoading: {d}")
        df = load_directory(d)
        df = assign_labels(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined):,} rows")
    print("\nRows per device:")
    print(combined['device_name'].value_counts().to_string())
    print("\nRows per category:")
    print(combined['category'].value_counts().to_string())

    combined = combined[combined['category'].notna()].copy()

    le_label = LabelEncoder()
    y = le_label.fit_transform(combined['category'].values)
    print(f"\nEncoded categories: {list(le_label.classes_)}")

    for set_name, feature_cols in FEATURE_SETS.items():
        print(f"\nFeature set '{set_name}' ({len(feature_cols)} requested):")
        X, encoders, used_cols = encode_features(combined, feature_cols, fit=True)
        print(f"Shape: {X.shape}")
        np.save(f'X_{set_name}.npy', X)
        joblib.dump(encoders, f'encoders_{set_name}.pkl')
        joblib.dump(used_cols, f'cols_{set_name}.pkl')

    np.save('y.npy', y)
    joblib.dump(le_label, 'label_encoder.pkl')
    np.save('device_names.npy', combined['device_name'].values)