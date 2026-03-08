"""Simple example of using the Combined ASD Data workbook for visualization
and training a scikit‑learn classifier.

Usage
-----
python3 train_asd_model.py /workspaces/tp_chemical_analysis/uploads/Combined_ASD_Data.xlsx

This script does the following:

1. load the "Reflectance_vs_Wavelength" sheet
2. parse the column names to extract a mineral label
3. build a matrix of samples (rows) x wavelengths (features)
4. train a RandomForestClassifier to predict the mineral from the
   reflectance curve (spectrum)
5. plot a few example spectra and an average spectrum per mineral

The point is not to provide a state‑of‑the‑art model but to show how the
workbook can be used as the sole source of truth for wavelengths and
reflectance values.  You can adapt the preprocessing, the choice of
model, or the plotting calls according to your needs.
"""

import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
# imbalanced learning utilities
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def extract_mineral(column_name: str) -> str:
    """Return the mineral name that appears in the ASD filename.

    The columns in the "Reflectance_vs_Wavelength" sheet are derived from
    the original file names, for example
    ``s07_ASD_Acmite_NMNH133746_Pyroxene_BECKa_AREF``.  The third
    underscore‑separated token is the mineral (``Acmite`` in this case).  The
    function is resilient to slightly different naming conventions by
    falling back to a regex when the simple split fails.
    """
    parts = column_name.split("_")
    if len(parts) >= 3:
        return parts[2]
    # try a regex fallback
    m = re.search(r"ASD_([A-Za-z0-9]+)", column_name)
    return m.group(1) if m else "unknown"


def load_dataset(xlsx_path: Path):
    df = pd.read_excel(xlsx_path, sheet_name="Reflectance_vs_Wavelength")
    # first column is 'Wavelength'; subsequent columns are samples
    wavelengths = df["Wavelength"].values
    samples = df.columns[1:]

    # build a sample x wavelength array
    data = df.iloc[:, 1:].T.values  # shape: (n_samples, n_wavelengths)
    minerals = [extract_mineral(col) for col in samples]

    # remove non‑positive wavelengths and corresponding columns
    valid_wl = wavelengths > 0
    wavelengths = wavelengths[valid_wl]
    data = data[:, valid_wl]

    # convert negative reflectance to NaN so later operations ignore them
    data = np.where(data < 0, np.nan, data)

    return data, minerals, wavelengths, samples, df


def plot_spectra(data, minerals, wavelengths, samples, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    # plot the first 50 spectra with color by mineral
    unique = sorted(set(minerals))
    colors = plt.cm.get_cmap("tab20", len(unique))
    label2color = {lab: colors(i) for i, lab in enumerate(unique)}

    plt.figure(figsize=(10, 6))
    for spec, lab in zip(data[:50], minerals[:50]):
        plt.plot(wavelengths, spec, color=label2color[lab], alpha=0.3)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Example spectra (first 50 samples)")
    plt.savefig(out_dir / "example_spectra.png")
    plt.close()

    # average spectrum per mineral
    avg = {}
    for lab in unique:
        avg[lab] = np.nanmean(data[np.array(minerals) == lab], axis=0)
    plt.figure(figsize=(10, 6))
    for lab, spec in avg.items():
        plt.plot(wavelengths, spec, label=lab)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.legend(loc="best", fontsize="small")
    plt.title("Average reflectance per mineral")
    plt.savefig(out_dir / "average_by_mineral.png")
    plt.close()

    # save average spectra to CSV for further use
    avg_df = pd.DataFrame({"Wavelength": wavelengths})
    for lab, spec in avg.items():
        avg_df[lab] = spec
    avg_df.to_csv(out_dir / "mean_spectra_per_mineral.csv", index=False)

    # plot all mean spectra together for quick inspection
    plt.figure(figsize=(10, 6))
    for lab, spec in avg.items():
        plt.plot(wavelengths, spec, label=lab, linewidth=0.8)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Mean reflectance spectra (all minerals)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="xx-small")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_spectra_all.png")
    plt.close()

    # also compute 'best' sample for each mineral (closest to mean)
    rep_df = pd.DataFrame({"Wavelength": wavelengths})
    for lab in unique:
        mask = np.array(minerals) == lab
        if mask.sum() == 0:
            continue
        spectra = data[mask]
        mean_spec = avg[lab]
        # compute L2 distance to mean
        dists = np.linalg.norm(spectra - mean_spec, axis=1)
        best_idx = np.argmin(dists)
        # pick the corresponding original column name
        sample_name = samples[np.where(mask)[0][best_idx]]
        rep_df[lab] = spectra[best_idx]
    rep_df.to_csv(out_dir / "representative_spectra.csv", index=False)

    return avg_df, rep_df


def train_model(data, minerals, out_dir: Path, raw_df=None,
                min_samples=5, top_n=None):
    # build DataFrame and count minerals
    df = pd.DataFrame({'spec': list(data), 'mineral': minerals})
    counts = df['mineral'].value_counts()

    # drop classes with too few examples
    valid = counts[counts >= min_samples].index
    if len(valid) < len(counts):
        print("Dropping minerals with <", min_samples, "samples:",
              list(counts[counts < min_samples].index))
    if top_n is not None:
        top = counts.loc[valid].nlargest(top_n).index
        valid = valid.intersection(top)
        print(f"Restricting to top {top_n} minerals by frequency")

    mask = df['mineral'].isin(valid)
    data_filt = np.vstack(df.loc[mask, 'spec'].values)
    minerals_filt = df.loc[mask, 'mineral'].tolist()

    print("Using the following minerals (counts):")
    print(counts.loc[valid])

    # engineer derivative features
    if raw_df is not None:
        wl = raw_df['Wavelength'].values
    else:
        wl = np.arange(data_filt.shape[1])
    deriv = np.gradient(data_filt, wl, axis=1)
    X = np.hstack([data_filt, deriv])

    # pipeline with imputation, scaling and oversampling
    from sklearn.impute import SimpleImputer
    from imblearn.over_sampling import RandomOverSampler
    pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("oversample", RandomOverSampler(random_state=42)),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "clf": [RandomForestClassifier(random_state=42),
                GradientBoostingClassifier(random_state=42)],
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 10, 20]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, minerals_filt, test_size=0.3, random_state=42, stratify=minerals_filt
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best parameters:", grid.best_params_)

    preds = best.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test accuracy:", acc)
    print(classification_report(y_test, preds, zero_division=0))

    # confusion matrix plot
    cm = confusion_matrix(y_test, preds, labels=best.classes_)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(range(len(best.classes_)), best.classes_, rotation=90, fontsize='small')
    plt.yticks(range(len(best.classes_)), best.classes_, fontsize='small')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png')
    plt.close()

    # PCA of engineered feature space
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(StandardScaler().fit_transform(X))
    plt.figure(figsize=(8, 6))
    for lab in sorted(set(minerals_filt)):
        mask2 = np.array(minerals_filt) == lab
        plt.scatter(X2[mask2, 0], X2[mask2, 1], label=lab, s=10, alpha=0.5)
    plt.legend(loc='best', fontsize='small')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA projection of spectra + derivative')
    plt.savefig(out_dir / 'pca.png')
    plt.close()

    joblib.dump(best, out_dir / 'best_model.joblib')
    print('Saved trained model to', out_dir / 'best_model.joblib')

    return best


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train_asd_model.py path/to/Combined_ASD_Data.xlsx")
        sys.exit(1)

    path = Path(sys.argv[1])
    data, minerals, wavelengths, samples, raw_df = load_dataset(path)
    print(f"loaded {data.shape[0]} samples with {data.shape[1]} effective wavelengths")

    out = Path("analysis_outputs")
    avg_df, rep_df = plot_spectra(data, minerals, wavelengths, samples, out)

    # train using at least 5 samples per class and restrict to top 50 minerals
    model = train_model(data, minerals, out, raw_df=raw_df,
                        min_samples=5, top_n=50)
    # the CSV files and model are in 'analysis_outputs'
    print("Training complete; plots, spectra and model saved to", out)
