"""LossRegularizer — handles all regularization terms, coefficient scheduling, and history tracking.

Extracted from models/utils.py for independent testability and clear module boundaries.
"""

import numpy as np
import torch

from flyvis_gnn.models.utils import get_in_features_g_phi, get_in_features_update
from flyvis_gnn.utils import to_numpy


class LossRegularizer:
    """
    Handles all regularization terms, coefficient scheduling, and history tracking.

    Usage:
        regularizer = LossRegularizer(train_config, model_config, activity_column=6,
                                       plot_frequency=100, n_neurons=1000, trainer_type='signal')

        for epoch in range(n_epochs):
            regularizer.set_epoch(epoch)

            for N in range(Niter):
                regularizer.reset_iteration()

                pred, in_features, msg = model(batch, data_id=data_id, return_all=True)

                regul_loss = regularizer.compute(model, x, in_features, ids, ids_batch, edges, device)
                loss = pred_loss + regul_loss
    """

    # Components tracked in history
    COMPONENTS = [
        'W_L1', 'W_L2', 'W_sign',
        'g_phi_diff', 'g_phi_norm', 'g_phi_weight', 'f_theta_weight',
        'f_theta_zero', 'f_theta_diff', 'f_theta_msg_diff', 'f_theta_msg_sign',
        'missing_activity', 'model_a', 'model_b', 'modulation',
        'f_theta_linearity', 'f_theta_centering'
    ]

    def __init__(self, train_config, model_config, activity_column: int,
                 plot_frequency: int, n_neurons: int, trainer_type: str = 'signal',
                 dataset: str = None):
        """
        Args:
            train_config: TrainingConfig with coeff_* values
            model_config: GraphModelConfig with model settings
            activity_column: Column index for activity (6 for signal, 3 for flyvis)
            plot_frequency: How often to record to history
            n_neurons: Number of neurons for normalization
            trainer_type: 'signal' or 'flyvis'
        """
        self.train_config = train_config
        self.model_config = model_config
        self.activity_column = activity_column
        self.plot_frequency = plot_frequency
        self.n_neurons = n_neurons
        self.trainer_type = trainer_type

        # Current epoch
        self.epoch = 0
        self.Niter = 0

        # Iteration counter
        self.iter_count = 0

        # Per-iteration accumulator
        self._iter_total = 0.0
        self._iter_tracker = {}

        # Epoch boundary tracking (cumulative iter_count at each epoch start)
        self.epoch_boundaries = []

        # History for plotting
        self._history = {comp: [] for comp in self.COMPONENTS}
        self._history['regul_total'] = []
        self._history['iteration'] = []

        # Cache coefficients
        self._coeffs = {}
        self._update_coeffs()

        # f_theta linearity loss state (unsupervised — no gt V_rest needed)
        self._mu_activity = None
        self._sigma_activity = None

    def set_activity_stats(self, x_ts, device):
        """Cache per-neuron activity statistics for linearity loss.

        Should be called once after construction, before training starts.

        Args:
            x_ts: NeuronTimeSeries with voltage field.
            device: Torch device.
        """
        from flyvis_gnn.metrics import compute_activity_stats
        mu, sigma = compute_activity_stats(x_ts, device)
        self._mu_activity = to_numpy(mu).astype(np.float32)
        self._sigma_activity = to_numpy(sigma).astype(np.float32)

    def _update_coeffs(self):
        """Recompute coefficients based on current epoch.

        Weight regularization (L1 and L2) uses exponential annealing:
        coeff * (1 - exp(-rate * epoch)), controlled by regul_annealing_rate.
        With rate=0.5 (default), ramps from 0 at epoch 0 to ~0.39x at epoch 1
        to ~0.92x at epoch 5. This allows the model to learn dynamics before
        regularization pressure is applied — critical for SIREN visual field training.
        """
        tc = self.train_config
        epoch = self.epoch
        rate = tc.regul_annealing_rate

        # Exponential ramp-up annealing for weight regularization
        def anneal(coeff):
            return coeff * (1 - np.exp(-rate * epoch)) if rate > 0 else coeff

        # Two-phase training support (like ParticleGraph data_train_synaptic2)
        n_epochs_init = getattr(tc, 'n_epochs_init', 0)
        first_coeff_L1 = getattr(tc, 'first_coeff_L1', tc.coeff_W_L1)

        if n_epochs_init > 0 and epoch < n_epochs_init:
            self._coeffs['W_L1'] = first_coeff_L1
        else:
            self._coeffs['W_L1'] = anneal(tc.coeff_W_L1)
        self._coeffs['W_L2'] = anneal(tc.coeff_W_L2)
        self._coeffs['g_phi_weight_L1'] = anneal(tc.coeff_g_phi_weight_L1)
        self._coeffs['g_phi_weight_L2'] = anneal(tc.coeff_g_phi_weight_L2)
        self._coeffs['f_theta_weight_L1'] = anneal(tc.coeff_f_theta_weight_L1)
        self._coeffs['f_theta_weight_L2'] = anneal(tc.coeff_f_theta_weight_L2)

        # Non-annealed coefficients
        self._coeffs['W_sign'] = tc.coeff_W_sign
        # Two-phase: g_phi_diff is active in phase 1, disabled in phase 2
        if n_epochs_init > 0 and epoch >= n_epochs_init:
            self._coeffs['g_phi_diff'] = 0  # Phase 2: no monotonicity constraint
        else:
            self._coeffs['g_phi_diff'] = tc.coeff_g_phi_diff
        self._coeffs['g_phi_norm'] = tc.coeff_g_phi_norm
        self._coeffs['f_theta_zero'] = tc.coeff_f_theta_zero
        self._coeffs['f_theta_diff'] = tc.coeff_f_theta_diff
        self._coeffs['f_theta_msg_diff'] = tc.coeff_f_theta_msg_diff
        self._coeffs['f_theta_msg_sign'] = tc.coeff_f_theta_msg_sign
        self._coeffs['missing_activity'] = tc.coeff_missing_activity
        self._coeffs['model_a'] = tc.coeff_model_a
        self._coeffs['model_b'] = tc.coeff_model_b
        self._coeffs['modulation'] = tc.coeff_lin_modulation
        self._coeffs['f_theta_linearity'] = getattr(tc, 'coeff_f_theta_linearity', 0.0)
        self._coeffs['f_theta_centering'] = getattr(tc, 'coeff_f_theta_centering', 0.0)

    def set_epoch(self, epoch: int, plot_frequency: int = None, Niter: int = None):
        """Set current epoch and update coefficients."""
        self.epoch = epoch
        self._update_coeffs()
        if plot_frequency is not None:
            self.plot_frequency = plot_frequency
        if Niter is not None:
            self.Niter = Niter
        if epoch > 0:
            self.epoch_boundaries.append(self.iter_count)

    def reset_iteration(self):
        """Reset per-iteration accumulator (called once per batch, NOT per N iteration)."""
        self._iter_total = 0.0
        self._iter_tracker = {comp: 0.0 for comp in self.COMPONENTS}
        # Flag to ensure W_L1 is only applied once per iteration (not per batch item)
        self._W_L1_applied_this_iter = False

    def should_record(self) -> bool:
        """Check if we should record to history this iteration."""
        return (self.iter_count % self.plot_frequency == 0) or (self.iter_count == 1)

    def needs_update_regul(self) -> bool:
        """Check if update regularization is needed (update_diff, update_msg_diff, or update_msg_sign)."""
        return (self._coeffs['f_theta_diff'] > 0 or
                self._coeffs['f_theta_msg_diff'] > 0 or
                self._coeffs['f_theta_msg_sign'] > 0)

    def _add(self, name: str, term):
        """Internal: accumulate a regularization term."""
        if term is None:
            return
        val = term.item() if hasattr(term, 'item') else float(term)
        self._iter_total += val
        if name in self._iter_tracker:
            self._iter_tracker[name] += val

    def compute(self, model, x, in_features, ids, ids_batch, edges, device,
                xnorm=1.0, index_weight=None):
        """
        Compute all regularization terms internally.

        Args:
            model: The neural network model
            x: NeuronState — only voltage is used
            in_features: Features for f_theta (from model forward pass, can be None)
            ids: Sample indices for regularization
            ids_batch: Batch indices
            edges: Edge tensor
            device: Torch device
            xnorm: Normalization value
            index_weight: Index for W_sign computation (signal only)

        Returns:
            Total regularization loss tensor
        """
        tc = self.train_config
        mc = self.model_config
        n_neurons = self.n_neurons
        total_regul = torch.tensor(0.0, device=device)

        # Get model W (handle multi-run case not working here)
        # For low_rank_factorization, compute W from WL @ WR to allow gradient flow

        # --- W regularization ---

        low_rank = getattr(model, 'low_rank_factorization', False)
        if low_rank and hasattr(model, 'WL') and hasattr(model, 'WR'):

            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = (model.WL.norm(1) + model.WR) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True
        else:

            # W_L1: Apply only once per iteration (not per batch item)
            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(1) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True

            if self._coeffs['W_L2'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(2) * self._coeffs['W_L2']
                total_regul = total_regul + regul_term
                self._add('W_L2', regul_term)

        # --- g_phi / f_theta weight regularization ---
        if (self._coeffs['g_phi_weight_L1'] + self._coeffs['g_phi_weight_L2']) > 0 and hasattr(model, 'g_phi'):
            for param in model.g_phi.parameters():
                regul_term = param.norm(1) * self._coeffs['g_phi_weight_L1'] + param.norm(2) * self._coeffs['g_phi_weight_L2']
                total_regul = total_regul + regul_term
                self._add('g_phi_weight', regul_term)

        if (self._coeffs['f_theta_weight_L1'] + self._coeffs['f_theta_weight_L2']) > 0 and hasattr(model, 'f_theta'):
            for param in model.f_theta.parameters():
                regul_term = param.norm(1) * self._coeffs['f_theta_weight_L1'] + param.norm(2) * self._coeffs['f_theta_weight_L2']
                total_regul = total_regul + regul_term
                self._add('f_theta_weight', regul_term)

        # --- f_theta_zero regularization ---
        if self._coeffs['f_theta_zero'] > 0 and hasattr(model, 'f_theta'):
            in_features_phi = get_in_features_update(rr=None, model=model, device=device)
            func_phi = model.f_theta(in_features_phi[ids].float())
            regul_term = func_phi.norm(2) * self._coeffs['f_theta_zero']
            total_regul = total_regul + regul_term
            self._add('f_theta_zero', regul_term)

        # --- g_phi diff/norm regularization ---
        if ((self._coeffs['g_phi_diff'] > 0) | (self._coeffs['g_phi_norm'] > 0)) and hasattr(model, 'g_phi'):
            in_features_edge, in_features_edge_next = get_in_features_g_phi(x, model, mc, xnorm, n_neurons, device)

            if self._coeffs['g_phi_diff'] > 0:
                if mc.g_phi_positive:
                    msg0 = model.g_phi(in_features_edge[ids].clone().detach()) ** 2
                    msg1 = model.g_phi(in_features_edge_next[ids].clone().detach()) ** 2
                else:
                    msg0 = model.g_phi(in_features_edge[ids].clone().detach())
                    msg1 = model.g_phi(in_features_edge_next[ids].clone().detach())
                regul_term = torch.relu(msg0 - msg1).norm(2) * self._coeffs['g_phi_diff']
                total_regul = total_regul + regul_term
                self._add('g_phi_diff', regul_term)

            if self._coeffs['g_phi_norm'] > 0:
                in_features_edge_norm = in_features_edge.clone()
                in_features_edge_norm[:, 0] = 2 * xnorm
                if mc.g_phi_positive:
                    msg_norm = model.g_phi(in_features_edge_norm[ids].clone().detach()) ** 2
                else:
                    msg_norm = model.g_phi(in_features_edge_norm[ids].clone().detach())
                # Different normalization target for signal vs flyvis
                if self.trainer_type == 'signal':
                    regul_term = (msg_norm - 1).norm(2) * self._coeffs['g_phi_norm']
                else:  # flyvis
                    regul_term = (msg_norm - 2 * xnorm).norm(2) * self._coeffs['g_phi_norm']
                total_regul = total_regul + regul_term
                self._add('g_phi_norm', regul_term)

        # --- W_sign (Dale's Law) regularization ---
        if self._coeffs['W_sign'] > 0 and self.epoch > 0:
            W_sign_temp = getattr(tc, 'W_sign_temperature', 10.0)

            if self.trainer_type == 'signal' and index_weight is not None:
                # Signal version: uses index_weight
                if self.iter_count % 4 == 0:
                    W_sign = torch.tanh(5 * model_W) # noqa: F821
                    loss_contribs = []
                    for i in range(n_neurons):
                        indices = index_weight[int(i)]
                        if indices.numel() > 0:
                            values = W_sign[indices, i]
                            std = torch.std(values, unbiased=False)
                            loss_contribs.append(std)
                    if loss_contribs:
                        regul_term = torch.stack(loss_contribs).norm(2) * self._coeffs['W_sign']
                        total_regul = total_regul + regul_term
                        self._add('W_sign', regul_term)
            else:
                # Flyvis version: uses scatter_add
                weights = model_W.squeeze() if model_W is not None else model.W.squeeze() # noqa: F821
                source_neurons = edges[0]

                n_pos = torch.zeros(n_neurons, device=device)
                n_neg = torch.zeros(n_neurons, device=device)
                n_total = torch.zeros(n_neurons, device=device)

                pos_mask = torch.sigmoid(W_sign_temp * weights)
                neg_mask = torch.sigmoid(-W_sign_temp * weights)

                n_pos.scatter_add_(0, source_neurons, pos_mask)
                n_neg.scatter_add_(0, source_neurons, neg_mask)
                n_total.scatter_add_(0, source_neurons, torch.ones_like(weights))

                violation = torch.where(n_total > 0,
                                        (n_pos / n_total) * (n_neg / n_total),
                                        torch.zeros_like(n_total))
                regul_term = violation.sum() * self._coeffs['W_sign']
                total_regul = total_regul + regul_term
                self._add('W_sign', regul_term)

        # Note: f_theta regularizations (f_theta_msg_diff, f_theta_msg_sign)
        # are handled by compute_update_regul() which should be called after the model forward pass.
        # Call finalize_iteration() after all regularizations are computed to record to history.

        # --- f_theta linearity loss (unsupervised, requires f_theta + a) ---
        if (self._coeffs['f_theta_linearity'] > 0
                and self._mu_activity is not None
                and hasattr(model, 'f_theta')):
            tc = self.train_config
            warmup_threshold = int(getattr(tc, 'f_theta_linearity_warmup_fraction', 0.3) * self.Niter)
            if self.iter_count > warmup_threshold:
                rampup_iters = getattr(tc, 'f_theta_linearity_rampup_iters', 200)
                rampup_weight = min(1.0, (self.iter_count - warmup_threshold) / max(rampup_iters, 1))

                from flyvis_gnn.metrics import compute_f_theta_linearity_loss
                lin_loss = compute_f_theta_linearity_loss(
                    model=model,
                    n_neurons=self.n_neurons,
                    mu=self._mu_activity,
                    sigma=self._sigma_activity,
                    device=device,
                )
                lin_term = lin_loss * self._coeffs['f_theta_linearity'] * rampup_weight
                total_regul = total_regul + lin_term
                self._add('f_theta_linearity', lin_term)

        # --- f_theta centering loss (unsupervised V_rest anchor, requires f_theta + a) ---
        if (self._coeffs['f_theta_centering'] > 0
                and self._mu_activity is not None
                and hasattr(model, 'f_theta')):
            tc = self.train_config
            warmup_threshold = int(
                getattr(tc, 'f_theta_centering_warmup_fraction', 0.3) * self.Niter)
            if self.iter_count > warmup_threshold:
                rampup_iters = getattr(tc, 'f_theta_centering_rampup_iters', 200)
                rampup_weight = min(
                    1.0,
                    (self.iter_count - warmup_threshold) / max(rampup_iters, 1))

                from flyvis_gnn.metrics import compute_f_theta_centering_loss
                cent_loss = compute_f_theta_centering_loss(
                    model=model,
                    n_neurons=self.n_neurons,
                    mu=self._mu_activity,
                    device=device,
                )
                cent_term = cent_loss * self._coeffs['f_theta_centering'] * rampup_weight
                total_regul = total_regul + cent_term
                self._add('f_theta_centering', cent_term)

        return total_regul

    def _record_to_history(self):
        """Append current iteration values to history."""
        n = self.n_neurons
        self._history['regul_total'].append(self._iter_total / n)
        self._history['iteration'].append(self.iter_count)
        for comp in self.COMPONENTS:
            self._history[comp].append(self._iter_tracker.get(comp, 0) / n)

    def compute_update_regul(self, model, in_features, ids_batch, device,
                              x=None, xnorm=None, ids=None):
        """
        Compute update function regularizations (f_theta_diff, f_theta_msg_diff, f_theta_msg_sign).

        This method should be called after the model forward pass when in_features is available.

        Args:
            model: The neural network model
            in_features: Features from model forward pass
            ids_batch: Batch indices
            device: Torch device
            x: Input tensor (required for update_diff with 'generic' update_type)
            xnorm: Normalization value (required for update_diff)
            ids: Sample indices (required for update_diff)

        Returns:
            Total update regularization loss tensor
        """
        mc = self.model_config
        embedding_dim = mc.embedding_dim
        n_neurons = self.n_neurons
        total_regul = torch.tensor(0.0, device=device)

        # update_diff: for 'generic' update_type only (requires g_phi, f_theta, a)
        if (self._coeffs['f_theta_diff'] > 0) and (model.update_type == 'generic') and (x is not None) and hasattr(model, 'f_theta'):
            in_features_edge, in_features_edge_next = get_in_features_g_phi(
                x, model, mc, xnorm, n_neurons, device)
            if mc.g_phi_positive:
                msg0 = model.g_phi(in_features_edge[ids].clone().detach()) ** 2
                msg1 = model.g_phi(in_features_edge_next[ids].clone().detach()) ** 2
            else:
                msg0 = model.g_phi(in_features_edge[ids].clone().detach())
                msg1 = model.g_phi(in_features_edge_next[ids].clone().detach())
            in_feature_update = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                           model.a[:n_neurons], msg0,
                                           torch.ones((n_neurons, 1), device=device)), dim=1)
            in_feature_update = in_feature_update[ids]
            in_feature_update_next = torch.cat((torch.zeros((n_neurons, 1), device=device),
                                                model.a[:n_neurons], msg1,
                                                torch.ones((n_neurons, 1), device=device)), dim=1)
            in_feature_update_next = in_feature_update_next[ids]
            regul_term = torch.relu(model.f_theta(in_feature_update) - model.f_theta(in_feature_update_next)).norm(2) * self._coeffs['f_theta_diff']
            total_regul = total_regul + regul_term
            self._add('f_theta_diff', regul_term)

        if in_features is None:
            return total_regul

        if self._coeffs['f_theta_msg_diff'] > 0:
            pred_msg = model.f_theta(in_features.clone().detach())
            in_features_msg_next = in_features.clone().detach()
            in_features_msg_next[:, embedding_dim + 1] = in_features_msg_next[:, embedding_dim + 1] * 1.05
            pred_msg_next = model.f_theta(in_features_msg_next)
            regul_term = torch.relu(pred_msg[ids_batch] - pred_msg_next[ids_batch]).norm(2) * self._coeffs['f_theta_msg_diff']
            total_regul = total_regul + regul_term
            self._add('f_theta_msg_diff', regul_term)

        if self._coeffs['f_theta_msg_sign'] > 0:
            in_features_modified = in_features.clone().detach()
            in_features_modified[:, 0] = 0
            pred_msg = model.f_theta(in_features_modified)
            msg_col = in_features[:, embedding_dim + 1].clone().detach()
            regul_term = (torch.tanh(pred_msg / 0.1) - torch.tanh(msg_col.unsqueeze(-1) / 0.1)).norm(2) * self._coeffs['f_theta_msg_sign']
            total_regul = total_regul + regul_term
            self._add('f_theta_msg_sign', regul_term)

        return total_regul

    def finalize_iteration(self):
        """
        Finalize the current iteration by recording to history if appropriate.

        This should be called once per training iteration N (after all batch regularizations).
        iter_count increments here — NOT in reset_iteration() — so it counts iterations, not batches.
        """
        self.iter_count += 1
        if self.should_record():
            self._record_to_history()

    def get_iteration_total(self) -> float:
        """Get total regularization for current iteration."""
        return self._iter_total

    def get_history(self) -> dict:
        """Get history dictionary for plotting."""
        return self._history
