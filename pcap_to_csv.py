import os
import glob
import argparse
import math
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import pyshark
from tqdm import tqdm


# NOTE: i believe some of this code may be reused or repurposed from a previous project,
# notably the previous semesters work with the DAMS lab. that repository is hidden,
# unfortunately, as the repo is not managed by me, but by the DAMS lab organization.


# OUI lookup  is first 3 bytes of MAC
# AI generated function
def get_oui(mac: str) -> str:
    if not mac or mac.lower() in ('', 'none', 'n/a'):
        return ''
    parts = mac.upper().replace('-', ':').split(':')
    if len(parts) < 3:
        return mac.upper()
    return ':'.join(parts[:3])

# check randomness/entropy
def payload_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = defaultdict(int)
    for b in data:
        freq[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)


# AI (claude, it's always claude) generated function
def stream_key(pkt) -> str:
    try:
        src = pkt.ip.src
        dst = pkt.ip.dst
        proto = pkt.transport_layer or 'other'
        sp = getattr(getattr(pkt, proto.lower(), None), 'srcport', '0') or '0'
        dp = getattr(getattr(pkt, proto.lower(), None), 'dstport', '0') or '0'
        a = (src, sp)
        b = (dst, dp)
        if a > b:
            a, b = b, a
        return f"{a[0]}:{a[1]}-{b[0]}:{b[1]}-{proto}"
    except AttributeError:
        return 'unknown'


# safe converting to int, recommended apparent;ly
def safe_int(val, default=0) -> int:

    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def extract_packet_features(pkt, device_mac: str) -> dict | None:

    try:
        ts = float(pkt.sniff_timestamp)
    except (AttributeError, ValueError):
        return None

    f = {
        'timestamp': ts,
        'stream': stream_key(pkt),

        # eth
        'src_mac': getattr(getattr(pkt, 'eth', None), 'src', ''),
        'dst_mac': getattr(getattr(pkt, 'eth', None), 'dst', ''),
        'eth_size': safe_int(getattr(getattr(pkt, 'eth', None), 'len', None)) or safe_int(getattr(pkt, 'length', 0)),
        # IP
        'src_ip': getattr(getattr(pkt, 'ip', None), 'src', ''),
        'dst_ip': getattr(getattr(pkt, 'ip', None), 'dst', ''),
        'ttl': safe_int(getattr(getattr(pkt, 'ip', None), 'ttl', None)),
        # trans
        'proto': pkt.transport_layer or 'other',
        'src_port': 0,
        'dst_port': 0,
        'l4_tcp': 0,
        'l4_udp': 0,
        # TCP flags
        'fin_flag': 0,
        'syn_flag': 0,
        'rst_flag': 0,
        'psh_flag': 0,
        'ack_flag': 0,
        'ece_flag': 0,
        'cwr_flag': 0,
        'tcp_window_size': 0,
        # app lyr
        'highest_layer': pkt.highest_layer,
        'payload_length': 0,
        'payload_entropy': 0.0,
        # TLS
        'handshake_version': 0,
        'handshake_cipher_suites_length': 0,
        'handshake_extensions_length': 0,
        'handshake_sig_hash_alg_len': 0,
        'tls_server': '',
        'handshake_ciphersuites': '',
        # HTTP
        'http_request_method': '',
        'http_host': '',
        'http_response_code': 0,
        'http_content_len': 0,
        'http_content_type': '',
        'http_uri': '',
        'user_agent': '',
        # DNS
        'dns_server': '',
        'dns_query_type': 0,
        'dns_len_qry': 0,
        'dns_len_ans': 0,
        'dns_interval': 0.0,
        # ICMP
        'icmp_type': 0,
        'icmp_checksum_status': 0,
        'icmp_data_size': 0,
        # identity
        'device_mac': device_mac,
        'eth_src_oui': '',
        'eth_dst_oui': '',
    }

    # OUI
    f['eth_src_oui'] = get_oui(f['src_mac'])
    f['eth_dst_oui'] = get_oui(f['dst_mac'])

    # TCP/UDP
    tcp = getattr(pkt, 'tcp', None)
    udp = getattr(pkt, 'udp', None)

    if tcp:
        f['l4_tcp'] = 1
        f['src_port'] = safe_int(tcp.srcport)
        f['dst_port'] = safe_int(tcp.dstport)
        f['tcp_window_size'] = safe_int(getattr(tcp, 'window_size_value', getattr(tcp, 'window', 0)))
        flags = safe_int(getattr(tcp, 'flags', '0x0'), 0)
        f['fin_flag'] = 1 if flags & 0x01 else 0
        f['syn_flag'] = 1 if flags & 0x02 else 0
        f['rst_flag'] = 1 if flags & 0x04 else 0
        f['psh_flag'] = 1 if flags & 0x08 else 0
        f['ack_flag'] = 1 if flags & 0x10 else 0
        f['ece_flag'] = 1 if flags & 0x40 else 0
        f['cwr_flag'] = 1 if flags & 0x80 else 0
        try:
            raw = bytes.fromhex(tcp.payload.replace(':', ''))
            f['payload_length'] = len(raw)
            f['payload_entropy'] = payload_entropy(raw)
        except (AttributeError, ValueError):
            pass

    elif udp:
        f['l4_udp'] = 1
        f['src_port'] = safe_int(udp.srcport)
        f['dst_port'] = safe_int(udp.dstport)
        try:
            raw = bytes.fromhex(udp.payload.replace(':', ''))
            f['payload_length'] = len(raw)
            f['payload_entropy'] = payload_entropy(raw)
        except (AttributeError, ValueError):
            pass

    # TLS
    tls = getattr(pkt, 'tls', None)
    if tls:
        f['handshake_version'] = safe_int(getattr(tls, 'handshake_version', 0))
        f['handshake_cipher_suites_length'] = safe_int(getattr(tls, 'handshake_ciphersuites_length', 0))
        f['handshake_extensions_length'] = safe_int(getattr(tls, 'handshake_extensions_length', 0))
        f['handshake_sig_hash_alg_len'] = safe_int(getattr(tls, 'handshake_sig_hash_alg_len', 0))
        f['tls_server'] = getattr(tls, 'handshake_extensions_server_name', '')
        f['handshake_ciphersuites'] = getattr(tls, 'handshake_ciphersuites', '')

    # HTTP
    http = getattr(pkt, 'http', None)
    if http:
        f['http_request_method'] = getattr(http, 'request_method', '')
        f['http_host'] = getattr(http, 'host', '')
        f['http_response_code'] = safe_int(getattr(http, 'response_code', 0))
        f['http_content_len'] = safe_int(getattr(http, 'content_length', 0))
        f['http_content_type'] = getattr(http, 'content_type', '')
        f['http_uri'] = getattr(http, 'request_uri', '')
        f['user_agent'] = getattr(http, 'user_agent', '')

    # DNS
    dns = getattr(pkt, 'dns', None)
    if dns:
        f['dns_server']     = f['dst_ip']
        f['dns_query_type'] = safe_int(getattr(dns, 'qry_type', 0))
        f['dns_len_qry']    = safe_int(getattr(dns, 'count_queries', 0))
        f['dns_len_ans']    = safe_int(getattr(dns, 'count_answers', 0))

    # ICMP
    icmp = getattr(pkt, 'icmp', None)
    if icmp:
        f['icmp_type']      = safe_int(getattr(icmp, 'type', 0))
        f['icmp_data_size'] = safe_int(getattr(icmp, 'data_len', 0))

    return f


def compute_window_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['jitter'] = df['timestamp'].diff().abs().fillna(0)

    # recommended - DONT CHANGE, IT'S FASTER!!!
    df['_stream_code'] = df['stream'].astype('category').cat.codes.astype(float)
    df['_src_ip_code'] = df['src_ip'].astype('category').cat.codes.astype(float)
    df['_channel']  = df['src_ip'] + '-' + df['dst_ip']
    df['_channel_code'] = df['_channel'].astype('category').cat.codes.astype(float)

    # recommended - DONT CHANGE, IT'S FASTER!!!
    dt_idx = pd.to_datetime(df['timestamp'], unit='s')
    df.index = dt_idx

    for w in windows:
        ws = f'{w}s'

        for col, prefix in [('_stream_code',  f'stream_{w}'),
                             ('_src_ip_code',  f'src_ip_{w}'),
                             ('_channel_code', f'channel_{w}')]:
            r = df[col].rolling(ws)
            df[f'{prefix}_count'] = r.apply(lambda x: len(set(x.astype(int))), raw=True).values
            df[f'{prefix}_mean'] = r.mean().values
            df[f'{prefix}_var'] = r.var().fillna(0).values

        jr = df['jitter'].rolling(ws)
        df[f'stream_jitter_{w}_sum'] = jr.sum().values
        df[f'stream_jitter_{w}_mean'] = jr.mean().values
        df[f'stream_jitter_{w}_var'] = jr.var().fillna(0).values

    df = df.drop(columns=['_stream_code', '_src_ip_code', '_channel', '_channel_code'])
    df = df.reset_index(drop=True)
    return df


def compute_packet_stats(df: pd.DataFrame) -> pd.DataFrame:
    # trying to match packet size stats matching CIC's
    # sum_p, min_p, max_p, med_p, average_p, var_p, q3_p, q1_p, iqr_p
    stats = df.groupby('stream')['eth_size'].agg(
        sum_p='sum', min_p='min', max_p='max',
        med_p='median', average_p='mean',
        var_p='var', q3_p=lambda x: x.quantile(0.75),
        q1_p=lambda x: x.quantile(0.25),
    ).reset_index()
    stats['iqr_p'] = stats['q3_p'] - stats['q1_p']
    stats['var_p'] = stats['var_p'].fillna(0)
    return df.merge(stats, on='stream', how='left')


def compute_timing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['inter_arrival_time'] = df['timestamp'].diff().fillna(0)
    df['time_since_previously_displayed_frame'] = df['inter_arrival_time']

    # NTP interval: mean time between UDP port 123 packets
    ntp_mask = (df['dst_port'] == 123) | (df['src_port'] == 123)
    if ntp_mask.sum() > 1:
        ntp_times = df.loc[ntp_mask, 'timestamp'].diff().dropna()
        ntp_val = float(ntp_times.mean())
    else:
        ntp_val = 0.0
    df['ntp_interval'] = ntp_val

    # DNS intrval: mean time between DNS packets per server
    dns_mask = (df['dst_port'] == 53) | (df['src_port'] == 53)
    if dns_mask.sum() > 1:
        dns_times = df.loc[dns_mask, 'timestamp'].diff().dropna()
        df.loc[dns_mask, 'dns_interval'] = float(dns_times.mean())

    return df


# misc features
def compute_misc(df: pd.DataFrame) -> pd.DataFrame:
    # most_freq_spot: dst_port of the most freq contacted port.
    if len(df) > 0:
        df['most_freq_spot'] = int(
            df['dst_port'].value_counts().idxmax()
        ) if df['dst_port'].nunique() > 0 else 0
    else:
        df['most_freq_spot'] = 0

    # l3_ip_dst_count: number of unique destination IPs seen so far (rolling aprox)
    seen = set()
    counts = []
    for ip in df['dst_ip']:
        seen.add(ip)
        counts.append(len(seen))
    df['l3_ip_dst_count'] = counts

    # port_class_dst: coarse port class (well-known/registered/dynamic).
    # should prolly by 1-hot instead?
    def port_class(p):
        if p < 1024: return 0 # well-known
        if p < 49152: return 1 # registered
        return 2 # dynamic or private or other
    df['port_class_dst'] = df['dst_port'].apply(port_class)

    # min_et, q1, min_e, var_, q1_e,
    ent_stats = df.groupby('stream')['payload_entropy'].agg(
        min_et='min',
        q1=lambda x: x.quantile(0.25),
        min_e='min',
        var_e='var',
        q1_e=lambda x: x.quantile(0.25),
    ).reset_index()
    ent_stats['var_e'] = ent_stats['var_e'].fillna(0)
    df = df.merge(ent_stats, on='stream', how='left', suffixes=('', '_ent'))
    return df

def process_pcap_dir(pcap_dir: str, device_name: str, device_mac: str, out_path: str, windows: list[int] = None):

    if windows is None:
        windows = [1, 5, 10, 30, 60]

    # collect & sort PCAPs in dir
    pcap_files = sorted(
        glob.glob(os.path.join(pcap_dir, '*.pcap'))
    )
    if not pcap_files:
        raise FileNotFoundError(f"No PCAP files found {pcap_dir}")

    print(f"Found {len(pcap_files)} PCAP file(s) in {pcap_dir}")
    print(f"Device: {device_name}, MAC is: {device_mac}")

    all_packets = []

    for pcap_file in tqdm(pcap_files, desc="Parsing PCAPs"):
        try:
            cap = pyshark.FileCapture(
                pcap_file,
                keep_packets=False,
                use_json=True,
                include_raw=False,
            )
            for pkt in cap:
                features = extract_packet_features(pkt, device_mac)
                if features:
                    all_packets.append(features)
            cap.close()
        except Exception as e:
            print(f"Warning: could not parse {pcap_file}: {e}")
            continue

    if not all_packets:
        raise ValueError("No packets successfully parsed.")

    print(f"\nParsed {len(all_packets):,} packets. Building feature matrix...")
    df = pd.DataFrame(all_packets)

    print("Computing timing features...")
    df = compute_timing(df)

    print("Computing time-window aggregations...")
    # subsamp. to 100k packets max to keep it RUNNABLE!
    if len(df) > 100_000:
        print(f"Subsampling from {len(df):,} to 100,000 packets for window computation...")
        df = df.sample(n=100_000, random_state = 42).reset_index(drop=True)

    df = compute_window_features(df, windows)

    print("Computing packet statistics...")
    df = compute_packet_stats(df)

    print("Computing miscellaneous features...")
    df = compute_misc(df)

    # assign device name as label (IMPORTANT: SAME AS CIC)
    df['device_mac'] = device_name.lower()

    # drop cols not in CIC
    drop_cols = ['timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert lab PCAP files to CIC-IoT-2024 feature CSVs'
    )
    parser.add_argument('--pcap_dir', required=True,
                        help='dir containing 1-hour PCAP files')
    parser.add_argument('--device_name', required=True,
                        help='device label (e.g. "google nest mini speaker")')
    parser.add_argument('--device_mac', required=True,
                        help='device MAC address (e.g. CC:F4:11:9C:D0:00)')
    parser.add_argument('--out', required=True,
                        help='output CSV path')
    args = parser.parse_args()

    process_pcap_dir(
        pcap_dir = args.pcap_dir,
        device_name = args.device_name,
        device_mac = args.device_mac,
        out_path = args.out,
    )