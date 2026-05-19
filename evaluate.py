import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample

FEATURE_SETS = ['full', 'no_id', 'behavioral']
MAX_PER_DEVICE = 10_000

# excluding Hub_Lighting!!! 0.00 F1 across all --> USELESS!!!
# feature sets due to Zigbee bridge architecture.
# TODO: report in writeup?
HELD_OUT_PER_CATEGORY = {
    'Audio': 'google nest mini speaker',
    'Camera': 'wyze camera',
    'Hub': 'aeotec smart home hub',
}


def load_everything(set_name):

    X = np.load(f'X_{set_name}.npy')
    y_raw = np.load('old/y.npy')
    device_names = np.load('old/device_names.npy', allow_pickle=True)
    le = joblib.load('old/label_encoder.pkl')
    cols = joblib.load(f'cols_{set_name}.pkl')
    y_float = np.array(y_raw, dtype=float)
    valid = ~np.isnan(y_float)
    return X[valid], y_float[valid].astype(int), device_names[valid], le, cols


def subsample_per_device(X, y, device_names, max_per_device):

    indices = []
    for device in np.unique(device_names):
        idx = np.where(device_names == device)[0]
        if len(idx) > max_per_device:
            idx = resample(idx, n_samples=max_per_device, random_state=42, replace=False)
        indices.append(idx)
    idx_all = np.concatenate(indices)
    return X[idx_all], y[idx_all], device_names[idx_all]


def leave_one_device_out(X, y, device_names, held_out_device):

    test_mask = device_names == held_out_device
    train_mask = ~test_mask
    return (X[train_mask], y[train_mask], X[test_mask],  y[test_mask])


# claude made this. anything that is using matploblib is likely claude, fed with inputs and outputs to create a function or chunk of code.
# i really do not like using matploblib. AI's do this menial task much faster
def plot_prediction_distribution(y_true, y_pred, label_encoder,
                                  held_out, set_name, filename):
    categories  = label_encoder.classes_
    n_cats      = len(categories)
    true_label  = label_encoder.inverse_transform([y_true[0]])[0]
    total       = len(y_pred)

    counts = np.array([
        (y_pred == i).sum() for i in range(n_cats)
    ])
    proportions = counts / total

    colors = ['#2ecc71' if categories[i] == true_label
              else '#e74c3c' for i in range(n_cats)]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(categories, proportions, color=colors)
    ax.set_ylabel('Proportion of predictions')
    ax.set_title(f'{set_name} | held out: {held_out}\n'
                 f'True label: {true_label} '
                 f'(n={total:,})', fontsize=9)
    ax.set_ylim(0, 1)

    for bar, prop in zip(bars, proportions):
        if prop > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    prop + 0.02, f'{prop:.2f}',
                    ha='center', va='bottom', fontsize=8)

    ax.axhline(y=1/n_cats, color='gray', linestyle='--',
               linewidth=0.8, label='Random chance')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# claude did this
def plot_multiclass_confusion(y_true, y_pred, label_encoder,
                               title, filename):
    all_labels = np.arange(len(label_encoder.classes_))
    cm         = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig, ax    = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# claude did this
def plot_importance_comparison(importances_dict, filename):
    n    = len(importances_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (set_name, imp) in zip(axes, importances_dict.items()):
        imp.sort_values().tail(15).plot(kind='barh', ax=ax,
                                        color='steelblue')
        ax.set_title(set_name)
        ax.set_xlabel('Importance')
    plt.suptitle('Top 15 features across ablation sets (RandomForest)',
                 y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


if __name__ == '__main__':

    label_encoder = joblib.load('old/label_encoder.pkl')
    categories = list(label_encoder.classes_)
    device_names_all = np.load('old/device_names.npy', allow_pickle=True)
    unique_devices = np.unique(device_names_all)
    test_devices = list(HELD_OUT_PER_CATEGORY.values())

    print(f"Categories: {categories}")
    print(f"Devices: {list(unique_devices)}\n")

    results = []
    importances = {}

    for set_name in FEATURE_SETS:
        print(f"Feature set: {set_name}")

        X, y, device_names, le, cols = load_everything(set_name)
        X, y, device_names = subsample_per_device(
            X, y, device_names, MAX_PER_DEVICE
        )

        for held_out in unique_devices:
            X_tr, y_tr, X_te, y_te = leave_one_device_out(
                X, y, device_names, held_out
            )
            if len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                continue

            dummy = DummyClassifier(strategy='stratified', random_state=42)
            dummy.fit(X_tr, y_tr)
            y_dummy = dummy.predict(X_te).astype(int)
            dummy_f1 = f1_score(y_te, y_dummy, average='macro', zero_division=0)
            dummy_acc = accuracy_score(y_te, y_dummy)

            rf = RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_te).astype(int)
            y_te = y_te.astype(int)

            f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
            acc = accuracy_score(y_te, y_pred)

            held_out_category = label_encoder.inverse_transform(
                [y_te[0]]
            )[0]

            # claude made this table-thing
            print(f"  {held_out:40s} | {held_out_category:12s} | "
                  f"F1: {f1:.4f} | Acc: {acc:.4f} | "
                  f"Dummy F1: {dummy_f1:.4f} | Dummy Acc: {dummy_acc:.4f}")

            plot_prediction_distribution(
                y_te, y_pred, label_encoder,
                held_out, set_name,
                f'pred_dist_{set_name}_{held_out.replace(" ","_")}.png'
            )

            results.append({
                'feature_set': set_name,
                'held_out_device': held_out,
                'held_out_category': held_out_category,
                'test_rows': len(X_te),
                'f1_macro': round(f1, 4),
                'accuracy': round(acc, 4),
                'dummy_f1': round(dummy_f1, 4),
                'dummy_acc': round(dummy_acc, 4),
            })

        print(f"\nGenerating multi-class confusion matrix [{set_name}]...")

        mc_test_mask = np.isin(device_names, test_devices)
        mc_train_mask = ~mc_test_mask

        rf_mc = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_mc.fit(X[mc_train_mask], y[mc_train_mask])
        importances[set_name] = pd.Series(
            rf_mc.feature_importances_, index=cols
        )
        y_pred_mc = rf_mc.predict(X[mc_test_mask]).astype(int)
        y_te_mc = y[mc_test_mask].astype(int)

        mc_f1 = f1_score(y_te_mc, y_pred_mc,
                          average='macro', zero_division=0)
        mc_acc = accuracy_score(y_te_mc, y_pred_mc)
        print(f"Multi-class F1 is: {mc_f1:.4f}, Acc is: {mc_acc:.4f}")
        print(classification_report(
            y_te_mc, y_pred_mc,
            target_names=label_encoder.classes_,
            zero_division=0
        ))

        # claude did this
        plot_multiclass_confusion(
            y_te_mc, y_pred_mc, label_encoder,
            title=f'Multi-class confusion -- {set_name}\n'
                  f'Train: all except held-out | '
                  f'Test: {", ".join(HELD_OUT_PER_CATEGORY.values())}',
            filename=f'cm_multiclass_{set_name}.png'
        )

    plot_importance_comparison(
        importances, 'old/importance_ablation_comparison.png'
    )

    df = pd.DataFrame(results)

    print("LEAVE-ONE-DEVICE-OUT RESULTS")

    pivot = df.pivot_table(
        index=['held_out_device', 'held_out_category'],
        columns='feature_set',
        values='f1_macro'
    ).reset_index()
    pivot['drop_from_full'] = (
        pivot.get('full', 0) - pivot.get('no_id', 0)
    ).round(4)
    print(pivot.to_string(index=False))

    pivot_acc = df.pivot_table(
        index=['held_out_device', 'held_out_category'],
        columns='feature_set',
        values='accuracy'
    ).reset_index()
    print("ACCURACY (bettr for single-class test sets)")
    print(pivot_acc.to_string(index=False))

    df.to_csv('lodo_results.csv', index=False)
    pivot.to_csv('ablation_pivot.csv', index=False)
    pivot_acc.to_csv('ablation_pivot_accuracy.csv', index=False)