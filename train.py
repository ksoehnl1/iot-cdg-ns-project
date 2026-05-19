import matplotlib
matplotlib.use('Agg')

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils import resample
from xgboost import XGBClassifier

FEATURE_SETS    = ['full', 'no_id', 'behavioral']
MAX_PER_DEVICE  = 10_000   # cap per device, not per category. REMEMBER!

# Hub_Lighting removed -- redundant bcuz we're testing cross-device
# Hub_Lighting has a 1 entry, practically useless
# also due to Zigbee bridge architecture incompatibility.
# could report separately as a negative result in the writeup.
# also other devices could be using Zigbee, must check
# very hard since don't have physical access and devices
# can be either or in some cases
HELD_OUT_PER_CATEGORY = {
    'Audio': 'google nest mini speaker',
    'Camera': 'wyze camera',
    'Hub': 'aeotec smart home hub',
}


def load_data(set_name):

    X  = np.load(f'X_{set_name}.npy')
    y_raw  = np.load('old/y.npy')
    device_names = np.load('old/device_names.npy', allow_pickle=True)
    le = joblib.load('old/label_encoder.pkl')
    cols = joblib.load(f'cols_{set_name}.pkl')

    # remove nan labels and cast to int once
    # both require integer labels; float labels cause random bs errors
    y_float      = np.array(y_raw, dtype=float)
    valid        = ~np.isnan(y_float)
    return (X[valid], y_float[valid].astype(int),
            device_names[valid], le, cols)


def subsample_per_device(X, y, device_names, max_per_device):

    indices = []
    for device in np.unique(device_names):
        idx = np.where(device_names == device)[0]
        if len(idx) > max_per_device:
            idx = resample(idx, n_samples=max_per_device, random_state=42, replace=False)
        indices.append(idx)
    idx_all = np.concatenate(indices)
    return X[idx_all], y[idx_all], device_names[idx_all]


def group_cross_validate(model, X, y, device_names, model_name, set_name):

    gkf = GroupKFold(n_splits=5) # KEEP. DONT USE STRATIFIED!
    scores = cross_val_score(
        model, X, y,
        cv=gkf.split(X, y, groups=device_names),
        scoring='f1_macro',
        error_score=np.nan   # KEEP OTHERWISE XGBOOST WILL THROW A FIT!!
    )
    if np.isnan(scores).all():
        print(f"[{set_name}] {model_name} GroupKFold CV: FAILED (missing classes in fold)")
        return np.nan
    print(f"[{set_name}] {model_name} GroupKFold CV F1: "
          f"{np.nanmean(scores):.4f} (+/- {np.nanstd(scores):.4f})")
    return np.nanmean(scores)


def dummy_baseline(X_train, y_train, X_test, y_test, set_name):

    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{set_name}] Dummy baseline -"
          f"F1 is: {f1:.4f} | Accuracy is: {acc:.4f}")
    return f1, acc

# claude made this
def plot_feature_importance(model, cols, model_name, set_name):
    imp = pd.Series(model.feature_importances_, index=cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, len(cols) // 4)))
    imp.tail(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Top 20 features — {model_name} [{set_name}]')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    fname = f'importance_{model_name}_{set_name}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")
    return imp


if __name__ == '__main__':

    label_encoder = joblib.load('old/label_encoder.pkl')
    print(f"Categories: {list(label_encoder.classes_)}")
    print(f"Held-out: {list(HELD_OUT_PER_CATEGORY.values())}")
    print(f"Max per device: {MAX_PER_DEVICE:,}\n")

    test_devices = list(HELD_OUT_PER_CATEGORY.values())
    summary = []

    for set_name in FEATURE_SETS:
        print(f"Feature set: {set_name}")

        X, y, device_names, le, cols = load_data(set_name)

        # subsample per device for balanced representation
        X, y, device_names = subsample_per_device(
            X, y, device_names, MAX_PER_DEVICE
        )
        print(f"After subsamp: {len(X):,} rows across "
              f"{len(np.unique(device_names))} devices")

        # device-aware split
        test_mask = np.isin(device_names, test_devices)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        device_names_train = device_names[train_mask]

        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

        if len(X_test) == 0:
            print("WARNING: NO TEST ROWS.")
            continue

        # dummy baseline: just (weighted) random guess
        dummy_f1, dummy_acc = dummy_baseline(
            X_train, y_train, X_test, y_test, set_name
        )

        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=1 # KEEP. NEED TO SHOW RESOURCES TO ESTIMATE TIME
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                verbosity=1 # SAME HERE
            ),
        }

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)

            cv_f1 = group_cross_validate(
                model, X_train, y_train,
                device_names_train, model_name, set_name
            )

            # Cross-device test
            y_pred = model.predict(X_test).astype(int)
            test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            test_acc = accuracy_score(y_test, y_pred)

            present_labels = np.unique(
                np.concatenate([y_test, y_pred])
            ).astype(int)
            present_names = label_encoder.inverse_transform(present_labels)

            # claude
            print(f"\n  Cross-device result "
                  f"(F1: {test_f1:.4f} | Acc: {test_acc:.4f} | "
                  f"Dummy F1: {dummy_f1:.4f}):")
            print(classification_report(
                y_test, y_pred,
                labels=present_labels,
                target_names=present_names,
                zero_division=0
            ))

            imp = plot_feature_importance(model, cols, model_name, set_name)
            print(f"Top 5 features:")
            print(imp.tail(5).to_string())

            joblib.dump(model, f'model_{model_name}_{set_name}.pkl')

            summary.append({
                'feature_set': set_name,
                'model': model_name,
                'dummy_f1': round(dummy_f1, 4),
                'cv_f1': round(cv_f1, 4),
                'crossdev_f1': round(test_f1, 4),
                'crossdev_acc': round(test_acc, 4),
                'train_rows': len(X_train),
                'test_rows': len(X_test),
            })

    print("ABLATION SUMMARY:")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    df_summary.to_csv('ablation_summary.csv', index=False)
    print("\nSaved: ablation_summary.csv")