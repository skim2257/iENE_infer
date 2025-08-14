from pathlib import Path
import os

import pandas as pd
from numpy import argsort
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve, precision_recall_curve

from matplotlib import pyplot as plt


def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test):
    return model.predict_proba(X_test)[:, 1]

def get_curves(y_true, y_hat):
    """
    returns the roc and precision-recall curves, x and y values.
    """
    fpr, tpr, _ = roc_curve(y_true, y_hat)
    precision, recall, _ = precision_recall_curve(y_true, y_hat)
    sort_idx = argsort(precision)

    return fpr, tpr, precision[sort_idx], recall[sort_idx]

def plot_curves(ax, y_true, y_hat, label):
    x_roc, y_roc, x_prc, y_prc = get_curves(y_true, y_hat)

    ax[0].plot(x_roc, y_roc, label=label)
    ax[1].plot(x_prc, y_prc, label=label)

    roc_auc = auc(x_roc, y_roc)
    pr_auc = auc(x_prc, y_prc)

    return roc_auc, pr_auc

def main():
    ### LOADING DATA ###
    code_dir = Path("C:/Users/samkr/OneDrive/Desktop/code/iENE_infer_new")
    df = pd.read_csv(code_dir / "src" / "notebooks" / "RADCURE_Clinical_v05.csv", index_col=0)

    # Load clinical data
    clinical_columns = ["Age", "Sex", "ECOG PS", "Smoking Status", "T", "N", "M", "HPV"]
    df_clin = pd.get_dummies(df[clinical_columns], drop_first=True)
    clinical_dumb_columns = df_clin.columns.tolist()
    combined_columns = ["vol_norm"] + clinical_dumb_columns
    print(combined_columns)

    # Load true iENE values
    df_true = pd.read_csv(Path(code_dir) / "iENE.csv", index_col=8).drop(columns=["Unnamed: 0"])

    # Merge clinical data with true iENE values
    df_merg = pd.merge(df_clin, df_true, left_index=True, right_index=True)

    # Print merged data
    print(df_merg.head(), df_merg.shape)

    # get the iENE predictions (heldout test set)
    df_pred = pd.read_csv(Path(code_dir) / os.environ["PRED_SAVE_PATH"].replace(".csv", "_AVERAGE.csv"))
    df_pred["ID"] = df_pred["ID"].apply(lambda x: x.split('/')[-1].split('_')[1])
    df_pred = df_pred.rename(columns={"ID": "patient_id"})


    ### TRAIN MODELS ###
    # Split data
    X_train = df_merg[df_merg["split"] == "train"]
    y_train = df_merg["ENE"][df_merg["split"] == "train"]
    X_test = df_merg[df_merg["split"] == "test"]
    y_test = df_merg["ENE"][df_merg["split"] == "test"]

    # Train models
    model_clin = train_model(X_train[clinical_dumb_columns], y_train)
    model_vol  = train_model(pd.DataFrame(X_train["vol_norm"]), y_train)
    model_comb = train_model(X_train[combined_columns], y_train)

    # Test models
    y_hat_clin = evaluate_model(model_clin, X_test[clinical_dumb_columns])
    y_hat_vol  = evaluate_model(model_vol, pd.DataFrame(X_test["vol_norm"]))
    y_hat_comb = evaluate_model(model_comb, X_test[combined_columns])

    # Plot curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    auroc_clin, auprc_clin = plot_curves(ax, y_test, y_hat_clin, "Clinical")
    auroc_vol, auprc_vol = plot_curves(ax, y_test, y_hat_vol, "Volume")
    auroc_comb, auprc_comb = plot_curves(ax, y_test, y_hat_comb, "Combined")


    # Add predictions
    df_pred_merg = pd.merge(df_pred, df_merg, left_on="patient_id", right_index=True, how="inner")
    y_hat_pred = df_pred_merg["ENE_average"]
    print(df_pred_merg.patient_id.nunique())
    df_group = df_pred_merg.groupby("patient_id").count()
    print(df_group[df_group["index"] > 1])
    auroc_pred, auprc_pred = plot_curves(ax, y_test, y_hat_pred, "Deep Learning")

    ax[0].legend()
    ax[1].legend()    

    # Print AUCs
    print(f"AUROC (clinical): {auroc_clin:.4f}, AUPRC (clinical): {auprc_clin:.4f}")
    print(f"AUROC (vol): {auroc_vol:.4f}, AUPRC (vol): {auprc_vol:.4f}")
    print(f"AUROC (comb): {auroc_comb:.4f}, AUPRC (comb): {auprc_comb:.4f}")
    print(f"AUROC (pred): {auroc_pred:.4f}, AUPRC (pred): {auprc_pred:.4f}")

    fig.savefig(code_dir / "outputs" / "figure1.png")
    plt.show()

if __name__ == "__main__":
    main()