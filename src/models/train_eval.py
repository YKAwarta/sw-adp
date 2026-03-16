"""
SW-ADP Stage 4: Shared training and evaluation for autoencoders.
Handles training loop, early stopping, anomaly scoring, and metric computation.
"""
import numpy as np
import torch
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, roc_auc_score)
from data_utils import masked_mse_loss, per_segment_masked_mse


def train_autoencoder(model, train_loader, val_loader, device,
                      epochs=100, lr=1e-3, patience=10, verbose=True):
    """
    Train an autoencoder (LSTM-AE or TCN-AE) with early stopping.

    Args:
        model: nn.Module with forward(x) -> reconstructed
        train_loader: DataLoader of (X, mask) normal segments
        val_loader: DataLoader of (X, mask) normal segments
        device: torch device
        epochs: max epochs
        lr: learning rate
        patience: early stopping patience
        verbose: print progress

    Returns:
        model: trained model (best weights)
        history: dict with train_loss, val_loss per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            X_batch, mask_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            reconstructed = model(X_batch)
            loss = masked_mse_loss(reconstructed, X_batch, mask_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X_batch, mask_batch = batch[0].to(device), batch[1].to(device)
                reconstructed = model(X_batch)
                loss = masked_mse_loss(reconstructed, X_batch, mask_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.1e}")

        # Early stopping
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (best val={best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history


def compute_anomaly_scores(model, dataloader, device):
    """
    Compute per-segment reconstruction error (anomaly score).

    Returns:
        scores: np.ndarray [N] reconstruction MSE per segment
        labels: np.ndarray [N] true labels (if available)
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                X_batch, mask_batch, y_batch = batch
                all_labels.append(y_batch.numpy())
            else:
                X_batch, mask_batch = batch

            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)

            reconstructed = model(X_batch)
            errors = per_segment_masked_mse(reconstructed, X_batch, mask_batch)
            all_scores.append(errors.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels) if all_labels else None
    return scores, labels


def evaluate_anomaly_detection(scores, y_true, contamination=0.197):
    """
    Evaluate anomaly detection using reconstruction error scores.

    Uses two approaches:
    1. Threshold at percentile matching contamination rate -> binary metrics
    2. Ranking metrics (AU-PR, ROC-AUC) that don't depend on threshold

    Returns:
        dict of metrics
    """
    # Threshold: top contamination% of scores are predicted anomalies
    threshold = np.percentile(scores, (1 - contamination) * 100)
    y_pred = (scores >= threshold).astype(int)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'au_pr': average_precision_score(y_true, scores),
        'roc_auc': roc_auc_score(y_true, scores),
    }


def run_ablation_for_model(model_class, model_kwargs_A, model_kwargs_B,
                           data, seeds, device, batch_size=32, epochs=100,
                           lr=1e-3, patience=10, model_name="Model"):
    """
    Run full ablation: train model under both conditions across multiple seeds.

    Args:
        model_class: class (LSTMAutoencoder or TCNAutoencoder)
        model_kwargs_A: dict of kwargs for Condition A (telem only)
        model_kwargs_B: dict of kwargs for Condition B (telem + SW)
        data: dict from load_data()
        seeds: list of random seeds
        device: torch device
        ...

    Returns:
        results_A: list of metric dicts (one per seed)
        results_B: list of metric dicts (one per seed)
        histories: dict of training histories
    """
    from data_utils import build_ae_dataloaders

    results = {}
    histories = {}

    for cond_label, X_key, model_kwargs in [
        ('Telem_Only', 'X_telem', model_kwargs_A),
        ('Telem_SW', 'X_telem_sw', model_kwargs_B),
    ]:
        print(f"\n{'='*60}")
        print(f"{model_name} - Condition: {cond_label}")
        print(f"{'='*60}")

        X = data[X_key]
        cond_results = []
        cond_histories = []

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Build dataloaders (re-split each seed for robustness)
            train_loader, val_loader, test_loader, test_y = build_ae_dataloaders(
                X, data['mask'], data['y'], data['train_mask'],
                batch_size=batch_size, val_frac=0.2, seed=seed
            )

            # Create fresh model
            model = model_class(**model_kwargs)
            n_params, size_kb = model.get_model_size()
            if seed == seeds[0]:
                print(f"  Model: {n_params:,} params ({size_kb:.1f} KB)")

            # Train
            t0 = time.time()
            model, history = train_autoencoder(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=lr, patience=patience, verbose=(seed == seeds[0])
            )
            elapsed = time.time() - t0
            print(f"  Training: {len(history['train_loss'])} epochs in {elapsed:.1f}s")
            cond_histories.append(history)

            # Evaluate
            scores, labels = compute_anomaly_scores(model, test_loader, device)
            contamination = data['y'][data['train_mask']].mean()
            metrics = evaluate_anomaly_detection(scores, labels, contamination)
            cond_results.append(metrics)
            print(f"  F1={metrics['f1']:.4f}  AU-PR={metrics['au_pr']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")

        results[cond_label] = cond_results
        histories[cond_label] = cond_histories

    return results, histories
