# Cross-Device IoT Traffic Classification

Code for "Cross-Device Generalization of IoT Traffic Classifiers: Vendor Ecosystems Dominate Over Device Categories" (CMSC 687, Spring 2025).

## What this does

Tests whether a traffic classifier trained on one IoT device can generalize to other devices in the same category. Short answer: no, but devices sharing a vendor ecosystem (e.g. Amazon Alexa) generalize well on behavioral features alone.

Uses leave-one-device-out evaluation with progressive feature ablation (full -> no_id -> behavioral) to show that vendor fingerprints, not behavioral signals, drive most of the classification performance.

## Dataset

CIC-IoT-2024 benign traffic, not included here.

https://www.unb.ca/cic/datasets/iot-diad-2024.html

Find and place the `BenignTraffic*.csv` files (not flow, but pcap features) in a directory like `CIC-IoT-2024/`.

## Run order

```
# 1. Preprocess the CIC data
python preprocess.py CIC-IoT-2024/

# 2. Train models and run ablation
python train.py

# 3. Evaluate with leave-one-device-out
python evaluate.py
```

For lab device evaluation (optional, requires own captured PCAPs):
```
# 4. Convert PCAPs to feature CSVs
python pcap_to_csv.py --pcap_dir ./captures/ --device_name "[name here]" --device_mac [mac here] --out output.csv

# 5. Evaluate lab data against trained models
python evaluate_dams.py
```

My own data was captured using the equipment in Dr. Roberto Yus' DAMS lab.

The PCAPs are not available (private lab data), but I have supplied the CSVs from the converted PCAP files in `DAMS-IoT-2026/`.

## Feature ablation sets

| Set | Description | Features |
|---|---|---|
| `full` | All 134 features | Includes MAC OUI, IPs, ports, DNS, TLS identifiers |
| `no_id` | No device identifiers | Strips MAC, IP, port, OUI, DNS server, TLS server, HTTP host/URI, user agent, cipher suites |
| `behavioral` | Traffic dynamics only | Timing, volume, jitter, packet size stats, so no deep packet inspection is needed |

The `behavioral` set is what a consumer router could realistically compute in real-time without DPI.

## Key results

- Device-aware GroupKFold CV: F1 = 0.16–0.41 (bad!)
- Amazon Alexa family (behavioral): 87–99% accuracy (good!)
- Google Nest Mini (behavioral): 39% accuracy (dummy baseline: 38%, so VERY bad!)
- Camera category: all was below random chance

See `results/ablation_pivot_accuracy.csv` and `results/lodo_results.csv` for full results.

## Files

| File | What it does |
|---|---|
| `preprocess.py` | Loads CIC CSVs, maps device labels, encodes features, saves .npy |
| `train.py` | Trains RF/XGBoost per feature set, runs GroupKFold CV, saves models |
| `evaluate.py` | Leave-one-device-out evaluation, confusion matrices, prediction distributions |
| `pcap_to_csv.py` | Converts lab PCAPs to CIC-compatible feature CSVs |
| `evaluate_lab.py` | Evaluates trained models on lab-captured data |
| `DAMS-IoT-2026/` | Feature CSVs preprocessed from DAMS lab |
| `results/` | Key results of the study, packaged into CSVs |

## Notes

- Hub_Lighting (Philips Hue Bridge) is excluded from evaluation; it's a single Zigbee bridge device that scores 0.00 across all feature sets due to architecture incompatibility with Wi-Fi direct devices.
- Some plotting functions, and other miscelleanous snippets were generated with AI assistance (sparsely noted in comments).
