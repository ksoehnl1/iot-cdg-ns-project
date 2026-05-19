"""
evaluates the lab captured device traffic against the classifers trained on CIC-IoT 2024 data.

usage is just:
    python evaluate_lab.py

edit LAB_DEVICES at the bottom to match whatever devices you're training

requirements:
    need to have model_RandomForest_*.pkl file (run train.py)
    encoders_*.pkl and cols_*.pkl must exist from running preprocess.py
    label_encoder.pkl must also exist from preprocess.py

    basically run this file last. the model needs to train on the CIC dataset before we can
    assess anything with the DAMS lab dataset
"""

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)

from preprocess import encode_features

FEATURE_SETS = ['full', 'no_id', 'behavioral']

# claude helped/made this plotting method
def plot_prediction_distribution(y_pred, true_category, device_name,
                                  label_encoder, set_name, filename):
    """
    Bar chart showing proportion of predictions per category.
    Green bar = correct category. Red bars = misclassifications.
    Dashed line = random chance baseline.
    """
    categories = label_encoder.classes_
    n_cats     = len(categories)
    total      = len(y_pred)

    counts      = np.array([(y_pred == i).sum() for i in range(n_cats)])
    proportions = counts / total
    colors      = ['#2ecc71' if categories[i] == true_category
                   else '#e74c3c' for i in range(n_cats)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars    = ax.bar(categories, proportions, color=colors, edgecolor='white')

    ax.set_ylabel('Proportion of predictions', fontsize=10)
    ax.set_title(
        f'Lab device: {device_name}\n'
        f'Feature set: {set_name} | True label: {true_category} '
        f'(n={total:,})',
        fontsize=9
    )
    ax.set_ylim(0, 1.1)

    # Label bars
    for bar, prop in zip(bars, proportions):
        if prop > 0.02:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                prop + 0.02,
                f'{prop:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    # Random chance line
    ax.axhline(
        y=1 / n_cats, color='gray', linestyle='--',
        linewidth=1.0, label=f'Random chance ({1/n_cats:.2f})'
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"    Saved: {filename}")

# claude helped/made this plotting method
def plot_summary_heatmap(df_results, filename):
    pivot = df_results.pivot_table(
        index='device', columns='feature_set', values='accuracy'
    )
    # Reorder columns if all three are present
    col_order = [c for c in ['full', 'no_id', 'behavioral']
                 if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(
        figsize=(max(5, len(col_order) * 2), max(3, len(pivot) * 0.8))
    )
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1,
                   aspect='auto')

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title('Lab device accuracy by feature set\n'
                 '(green = correct, red = wrong)', fontsize=10)

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='black' if 0.3 < val < 0.8 else 'white')

    plt.colorbar(im, ax=ax, label='Accuracy')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def evaluate_lab_device(csv_path, device_name, true_category,
                         label_encoder, set_name):

    df = pd.read_csv(csv_path, low_memory=False)

    encoders = joblib.load(f'encoders_{set_name}.pkl')
    feat_cols = joblib.load(f'cols_{set_name}.pkl')

    # NOTE: refitting remaps category integers and produces garbage predictions!!!
    X_lab, _, _ = encode_features(
        df, feat_cols, encoders=encoders, fit=False
    )

    if len(X_lab) == 0:
        print(f"WARNING: No rows after encoding for {device_name}")
        return None

    # evry row from this device has the same true label
    true_int = label_encoder.transform([true_category])[0]
    y_true = np.full(len(X_lab), true_int, dtype=int)

    model_path = f'model_RandomForest_{set_name}.pkl'
    if not os.path.exists(model_path):
        print(f"WARNING: {model_path} not found. Run train.py first.")
        return None

    model = joblib.load(model_path)
    y_pred = model.predict(X_lab).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    categories = label_encoder.classes_
    pred_counts = {
        cat: int((y_pred == i).sum())
        for i, cat in enumerate(categories)
    }
    pred_props = {
        cat: round(count / len(y_pred), 3)
        for cat, count in pred_counts.items()
    }

    print(f"Accuracy is : {acc:.4f} | F1: {f1:.4f}")
    print(f"Predictions are: { {k: v for k, v in pred_props.items() if v > 0} }")

    # plot stuff
    safe_name = device_name.replace(' ', '_')
    plot_prediction_distribution(
        y_pred, true_category, device_name,
        label_encoder, set_name,
        f'pred_dist_lab_{set_name}_{safe_name}.png'
    )

    return {
        'device': device_name,
        'category': true_category,
        'feature_set': set_name,
        'rows': len(X_lab),
        'accuracy': round(acc, 4),
        'f1_macro': round(f1, 4),
        **{f'pred_{cat}': pred_props.get(cat, 0.0)
           for cat in categories},
    }

if __name__ == '__main__':

    LAB_DEVICES = {
        'lab google nest mini': {
            'csv': './DAMS-IoT-2026/captures_csv/dams_google_nest_mini.csv',
            'category': 'Audio',
        },
        'lab amazon basics light': {
            'csv': './DAMS-IoT-2026/captures_csv/dams_amazon_basics_light.csv',
            'category': 'Hub_Lighting',
        },
        'lab amazon echo dot': {
            'csv': './DAMS-IoT-2026/captures_csv/dams_amazon_echo_dot.csv',
            'category': 'Audio',
        },
    }

    label_encoder = joblib.load('label_encoder.pkl')
    print(f"Categories: {list(label_encoder.classes_)}")
    print(f"Lab devices: {list(LAB_DEVICES.keys())}\n")

    all_results = []

    for set_name in FEATURE_SETS:
        print(f"Feature set: {set_name}")

        for device_name, info in LAB_DEVICES.items():
            if not os.path.exists(info['csv']):
                print(f"Skipping '{device_name}' - "
                      f"file not found: {info['csv']}")
                continue

            if info['category'] not in label_encoder.classes_:
                print(f"Skipping '{device_name}' - "
                      f"category '{info['category']}' not in th label encoder. "
                      f"Known: {list(label_encoder.classes_)}")
                continue

            print(f"\n[{device_name}]")
            result = evaluate_lab_device(
                csv_path= info['csv'],
                device_name = device_name,
                true_category = info['category'],
                label_encoder = label_encoder,
                set_name = set_name,
            )
            if result:
                all_results.append(result)

    if not all_results:
        print("\nNo results -- CHECK CSV PATHS IN LAB_DEVICES!!!")
    else:
        df = pd.DataFrame(all_results)

        print("LAB EVALUATION SUMMARY")

        summary_cols = ['device', 'category', 'feature_set',
                        'rows', 'accuracy', 'f1_macro']
        print(df[summary_cols].to_string(index=False))

        print("ACCURACY BY DEVICE × FEATURE SET")
        pivot = df.pivot_table(
            index= ['device', 'category'],
            columns = 'feature_set',
            values = 'accuracy'
        ).reset_index()
        col_order = ['device', 'category'] + [
            c for c in ['full', 'no_id', 'behavioral']
            if c in pivot.columns
        ]
        print(pivot[col_order].to_string(index=False))

        if 'full' in pivot.columns and 'no_id' in pivot.columns:
            pivot['drop_full_to_no_id'] = (
                pivot['full'] - pivot['no_id']
            ).round(4)
            print(f"\nDrop in accuracy from full -> no_id "
                  f"(vendor feature reliance):")
            print(pivot[['device', 'drop_full_to_no_id']].to_string(
                index=False
            ))

        df.to_csv('lab_results.csv', index=False)
        pivot.to_csv('lab_results_pivot.csv', index=False)
        plot_summary_heatmap(df, 'lab_accuracy_heatmap.png')