#!/usr/bin/env python

import os, sys
sys.path.insert(0, os.path.abspath('../../poplar'))
from poplar.nn.networks import LinearModel
from poplar.nn.training import train, train_test_split
from poplar.nn.rescaling import ZScoreRescaler
import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--events", type=str, required=True, help="total events")
parser.add_argument("--x_data_loc", type=str, required=True)
parser.add_argument("--y_data_loc", type=str, required=True)
parser.add_argument("--num_neurons", type=int, required=True)
parser.add_argument("--layers", type=int, required=True)
parser.add_argument("--train_test_frac", type=float, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--n_epochs", type=int, required=True)
parser.add_argument("--n_batches", type=int, required=True)
parser.add_argument("--update_every", type=int, required=True)
parser.add_argument("--loss_type", type=str, default="SmoothL1")
parser.add_argument("--rescaler_flag", type=str, default="use_rescaler")
parser.add_argument("--verbose", type=bool, required=True)
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--SURVEY", type=str, required=True)

args = parser.parse_args()

events_for_dir = f"{args.events}".replace("+0", "").replace(".0", "")
output_dir = f'./data/PRE_TRAIN_DATA_TDE/{events_for_dir}_events'

device = args.device

# load x data
x = np.load(args.x_data_loc)

# load y data
y = np.load(args.y_data_loc)

xdata = torch.as_tensor(x, device=device).float()
ydata = torch.as_tensor(y, device=device).float().view(-1)

ydata = ydata.float().view(-1)

print("Input Shape : ", xdata.shape, "Output shape : ", ydata.shape)

if args.rescaler_flag == "use_rescaler":
    if args.loss_type == "BCE":
        # X is pre-scaled; Y must stay in {0,1} — no rescaling needed at all
        rescaler = None
        print("WARNING: BCE loss detected — ignoring rescaler flag to protect labels.")
    else:
        # Regression: X already scaled, but rescaling Y (continuous) is still useful
        rescaler = ZScoreRescaler(xdata, ydata)
else:
    rescaler = None

model = LinearModel(
    in_features=xdata.shape[-1],
    out_features=1,
    neurons=[args.num_neurons] * args.layers ,
    activation=torch.nn.SiLU,
    rescaler=rescaler,
    dropout=0.1,
)

# no shuffling is required since this is random data 

model.set_device(device)

xtrain, xtest, ytrain, ytest = train_test_split([xdata, ydata], args.train_test_frac)  
optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=10, factor=0.5)

if args.loss_type == "MSE":
    loss_fn = torch.nn.MSELoss()
elif args.loss_type == "SmoothL1":
    loss_fn = torch.nn.SmoothL1Loss()
elif args.loss_type == "BCE":
    loss_fn = torch.nn.BCEWithLogitsLoss()
else:
    raise ValueError(f"Unsupported loss type: {args.loss_type}")

train(
    model, 
    data=[xtrain, ytrain, xtest, ytest], 
    n_epochs=args.n_epochs, 
    n_batches=args.n_batches, 
    loss_function=loss_fn,
    optimiser=optimiser,
    update_every=args.update_every,
    verbose=args.verbose,
    save_best=True,
    scheduler=scheduler,
    outdir=f'{output_dir}/{args.outdir}_{args.num_neurons}_{args.layers}_{args.n_batches}_{args.SURVEY}/',
)

# check errors on the model

ypred = model.run_on_dataset(xtest)

# For binary classification with BCE loss, apply sigmoid to logits
# if args.loss_type == "BCE":
#     ypred_sigmoid = torch.sigmoid(ypred)
#     # Compute binary accuracy
#     ypred_binary = (ypred_sigmoid > 0.5).float()
#     accuracy = (ypred_binary == ytest).float().mean().item()
#     print(f"Binary Accuracy: {accuracy:.4f}")
    
#     # Save predictions and ground truth
#     np.save(f'{output_dir}/{args.outdir}_{args.num_neurons}_{args.layers}_{args.n_batches}_{args.SURVEY}/predictions.npy', ypred_sigmoid.cpu().detach().numpy())
#     np.save(f'{output_dir}/{args.outdir}_{args.num_neurons}_{args.layers}_{args.n_batches}_{args.SURVEY}/ground_truth.npy', ytest.cpu().numpy())
    
#     # Plot ROC-like curve: prediction distribution by class
#     plt.figure(figsize=(10, 6))
#     plt.hist(ypred_sigmoid[ytest == 0].cpu().detach().numpy(), bins=30, alpha=0.6, label='Non-detectable (y=0)')
#     plt.hist(ypred_sigmoid[ytest == 1].cpu().detach().numpy(), bins=30, alpha=0.6, label='Detectable (y=1)')
#     plt.xlabel('Predicted Probability')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.savefig(f'{output_dir}/{args.outdir}_{args.num_neurons}_{args.layers}_{args.n_batches}_{args.SURVEY}/prediction_distribution.png', dpi=200)
#     plt.close()
# else:
#     plt.hist(np.log10(abs((1 - ypred/ytest).cpu().numpy())), bins='auto', density=True)
#     plt.xlabel('log_10 percent error')
#     plt.savefig(f'{output_dir}/{args.outdir}_{args.num_neurons}_{args.layers}_{args.n_batches}_{args.SURVEY}/percent_error.png', dpi=200)
#     plt.close()


# ── Evaluation & Plots ────────────────────────────────────────────────────
import os
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)

outpath = (
    f'{output_dir}/{args.outdir}_{args.num_neurons}'
    f'_{args.layers}_{args.n_batches}_{args.SURVEY}/'
)
os.makedirs(outpath, exist_ok=True)

ypred = model.run_on_dataset(xtest)

if args.loss_type == "BCE":

    probs  = torch.sigmoid(ypred).cpu().detach().numpy().ravel()
    labels = ytest.cpu().numpy().ravel().astype(int)

    np.save(outpath + 'predictions.npy', probs)
    np.save(outpath + 'ground_truth.npy', labels)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── 1. Predicted probability distribution by class ────────────────────
    ax = axes[0]
    ax.hist(probs[labels == 0], bins=40, alpha=0.6,
            color='steelblue', label='Non-detectable (y=0)', density=True)
    ax.hist(probs[labels == 1], bins=40, alpha=0.6,
            color='tomato',    label='Detectable (y=1)',     density=True)
    ax.axvline(0.5, ls='--', color='k', lw=1, label='threshold=0.5')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()

    # ── 2. ROC curve ──────────────────────────────────────────────────────
    ax = axes[1]
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()

    # ── 3. Precision–Recall curve ─────────────────────────────────────────
    ax = axes[2]
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap            = average_precision_score(labels, probs)
    ax.plot(rec, prec, color='mediumseagreen', lw=2,
            label=f'AP = {ap:.3f}')
    baseline = labels.mean()
    ax.axhline(baseline, ls='--', color='k', lw=1,
               label=f'Baseline (prevalence={baseline:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curve')
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath + 'classification_diagnostics.png', dpi=200)
    plt.close()

    # ── 4. Confusion matrix at 0.5 ───────────────────────────────────────
    preds_binary = (probs >= 0.5).astype(int)
    cm  = confusion_matrix(labels, preds_binary)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Non-det', 'Det']).plot(ax=ax)
    ax.set_title('Confusion Matrix (threshold=0.5)')
    plt.tight_layout()
    plt.savefig(outpath + 'confusion_matrix.png', dpi=200)
    plt.close()

    accuracy = (preds_binary == labels).mean()
    print(f"Binary Accuracy : {accuracy:.4f}")
    print(f"ROC AUC         : {roc_auc:.4f}")
    print(f"Avg Precision   : {ap:.4f}")

else:
    # ── Regression: predicted vs true scatter + percent-error hist ────────
    ypred_np = ypred.cpu().detach().numpy().ravel()
    ytest_np = ytest.cpu().numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: predicted vs true
    ax = axes[0]
    lims = [min(ytest_np.min(), ypred_np.min()),
            max(ytest_np.max(), ypred_np.max())]
    ax.scatter(ytest_np, ypred_np, s=2, alpha=0.3, color='steelblue')
    ax.plot(lims, lims, 'r--', lw=1, label='Perfect prediction')
    ax.set_xlabel('True value')
    ax.set_ylabel('Predicted value')
    ax.set_title('Predicted vs True (test set)')
    ax.legend()

    # Percent-error histogram
    ax = axes[1]
    pct_err = np.log10(np.abs(1 - ypred_np / ytest_np) + 1e-12)
    ax.hist(pct_err, bins='auto', density=True, color='steelblue')
    ax.set_xlabel('log₁₀ |percent error|')
    ax.set_ylabel('Density')
    ax.set_title('Percent Error Distribution')

    plt.tight_layout()
    plt.savefig(outpath + 'regression_diagnostics.png', dpi=200)
    plt.close()


# ── Direct True vs Predicted comparison ──────────────────────────────────

probs  = torch.sigmoid(ypred).cpu().detach().numpy().ravel()
labels = ytest.cpu().numpy().ravel().astype(int)

# Sort by predicted probability so the plot reads cleanly
sort_idx    = np.argsort(probs)
probs_sort  = probs[sort_idx]
labels_sort = labels[sort_idx]
n_test      = len(probs_sort)
sample_idx  = np.arange(n_test)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 1. Core "true vs predicted" plot ─────────────────────────────────────
ax = axes[0, 0]
# True labels as coloured markers (jittered vertically for visibility)
rng   = np.random.default_rng(42)
jitter = rng.uniform(-0.03, 0.03, size=n_test)

ax.scatter(sample_idx[labels_sort == 0], 
           labels_sort[labels_sort == 0] + jitter[labels_sort == 0],
           s=0.5, alpha=0.3, color='steelblue',  label='True: Non-det (0)')
ax.scatter(sample_idx[labels_sort == 1], 
           labels_sort[labels_sort == 1] + jitter[labels_sort == 1],
           s=0.5, alpha=0.3, color='tomato',     label='True: Det (1)')
# Predicted probability as a line
ax.plot(sample_idx, probs_sort, color='black', lw=0.8, 
        alpha=0.7, label='Predicted probability')
ax.axhline(0.5, ls='--', color='gray', lw=1, label='Threshold = 0.5')
ax.set_xlabel('Test sample index (sorted by predicted probability)')
ax.set_ylabel('Value')
ax.set_title('True Label vs Predicted Probability\n(sorted by predicted score)')
ax.legend(markerscale=8, fontsize=8)

# ── 2. Zoom into the decision boundary region ─────────────────────────────
ax = axes[0, 1]
# Only show samples where model is uncertain: predicted prob in [0.2, 0.8]
uncertain_mask = (probs_sort > 0.2) & (probs_sort < 0.8)
idx_unc   = sample_idx[uncertain_mask]
prob_unc  = probs_sort[uncertain_mask]
label_unc = labels_sort[uncertain_mask]

ax.scatter(idx_unc[label_unc == 0], prob_unc[label_unc == 0],
           s=4, alpha=0.5, color='steelblue', label='True: Non-det')
ax.scatter(idx_unc[label_unc == 1], prob_unc[label_unc == 1],
           s=4, alpha=0.5, color='tomato',    label='True: Det')
ax.axhline(0.5, ls='--', color='gray', lw=1)
ax.set_xlabel('Test sample index (sorted by predicted probability)')
ax.set_ylabel('Predicted probability')
ax.set_title('Decision Boundary Region [0.2 – 0.8]\n(where model is uncertain)')
ax.legend(fontsize=8)

# ── 3. Calibration curve (reliability diagram) ───────────────────────────
# This answers: "when model says 70%, is 70% of the time correct?"
ax = axes[1, 0]
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(labels, probs, n_bins=20)
ax.plot(prob_pred, prob_true, 's-', color='darkorange', lw=2, label='Model')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax.fill_between(prob_pred, prob_true, prob_pred,
                alpha=0.15, color='red', label='Calibration error')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Actual fraction of detectable events')
ax.set_title('Calibration Curve\n(does predicted probability match reality?)')
ax.legend(fontsize=8)

# ── 4. Per-bin accuracy across the probability range ─────────────────────
ax = axes[1, 1]
bins      = np.linspace(0, 1, 21)
bin_idx   = np.digitize(probs, bins) - 1
bin_acc   = []
bin_count = []
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for b in range(len(bins) - 1):
    mask = bin_idx == b
    if mask.sum() > 0:
        preds_b = (probs[mask] >= 0.5).astype(int)
        bin_acc.append((preds_b == labels[mask]).mean())
        bin_count.append(mask.sum())
    else:
        bin_acc.append(np.nan)
        bin_count.append(0)

bin_acc   = np.array(bin_acc)
bin_count = np.array(bin_count)

bars = ax.bar(bin_centers, bin_acc, width=0.045,
              color='mediumseagreen', alpha=0.7, label='Accuracy per bin')
ax2 = ax.twinx()
ax2.plot(bin_centers, bin_count, 'o--', color='navy',
         ms=4, lw=1, label='N samples per bin')
ax2.set_ylabel('Number of samples', color='navy')
ax.axhline(0.5, ls='--', color='gray', lw=1)
ax.set_xlabel('Predicted probability bin')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy and Sample Count per Probability Bin')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower left', fontsize=8)
ax2.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(outpath + 'true_vs_predicted.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved true_vs_predicted.png to {outpath}")