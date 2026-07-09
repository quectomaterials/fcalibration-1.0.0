import numpy as np
from sklearn.metrics import log_loss

class FcalibrationClass():
    def __init__(self):
        self.params = np.array([0.0, 0.0, 0.0, 0.0])

    def predicted_value(self, m, n):
        max_mn = np.maximum(m, n)
        min_mn = np.minimum(m, n)
        max_1m1n = np.maximum(1 - m, 1 - n)
        min_1m1n = np.minimum(1 - m, 1 - n)

        sig1 = 1.0 / (1.0 + np.exp(self.params[0] * min_mn + self.params[1]))
        sig2 = 1.0 / (1.0 + np.exp(self.params[2] * max_1m1n + self.params[3]))
        
        term1 = max_mn * sig1
        term2 = min_1m1n * sig2
        
        res = (term1 - term2 + 1.0) / 2.0
        return np.clip(res, 1e-15, 1 - 1e-15)

    def _compute_gradients(self, m, y, n, l2_reg):
        max_mn = np.maximum(m, n)
        min_mn = np.minimum(m, n)
        max_1m1n = np.maximum(1 - m, 1 - n)
        min_1m1n = np.minimum(1 - m, 1 - n)

        z1 = self.params[0] * min_mn + self.params[1]
        z2 = self.params[2] * max_1m1n + self.params[3]

        sig1 = 1.0 / (1.0 + np.exp(z1))
        sig2 = 1.0 / (1.0 + np.exp(z2))
        
        y_pred = (max_mn * sig1 - min_1m1n * sig2 + 1.0) / 2.0
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        dLoss_dy = (y_pred_clipped - y) / (y_pred_clipped * (1.0 - y_pred_clipped))
        
        dLoss_dz1 = dLoss_dy * (-0.5 * max_mn * sig1 * (1.0 - sig1))
        dLoss_dz2 = dLoss_dy * (0.5 * min_1m1n * sig2 * (1.0 - sig2))
        
        grads = np.array([
            np.mean(dLoss_dz1 * min_mn) + l2_reg * self.params[0],
            np.mean(dLoss_dz1),
            np.mean(dLoss_dz2 * max_1m1n) + l2_reg * self.params[2],
            np.mean(dLoss_dz2)
        ])
        return grads

    def fit(self, m_train, n_train, y_train, m_val=None, n_val=None, y_val=None,
            learning_rate=0.5, epochs=5000, smoothing=0.0, patience=20, 
            lr_patience=5, factor=0.5, min_lr=1e-5, l2_reg=1e-4, verbose=True):
        
        m_train = np.array(m_train)
        n_train = np.array(n_train)
        y_train = np.array(y_train)
        
        prior0 = np.mean(y_train <= 0)
        prior1 = np.mean(y_train > 0)
        
        if smoothing > 0:
            xp_agreement = 1 - np.abs(m_train - n_train)
            agreement = np.where(y_train > 0, n_train, 1 - n_train)
            trust = agreement / (xp_agreement + 1e-16)
            smoothing_per_sample = smoothing * (1 - trust)
            t = y_train * (1 - smoothing_per_sample) + 0.5 * smoothing_per_sample
        else:
            t = y_train
            
        initial_bias = float(np.log((prior1 + 1e-5) / (prior0 + 1e-5)))
        
        if self.params[1] == 0.0: self.params[1] = initial_bias
        if self.params[3] == 0.0: self.params[3] = initial_bias
        
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        w = np.zeros(4)
        v = np.zeros(4)
        
        best_loss = np.inf
        patience_counter = 0
        lr_patience_counter = 0  
        best_params = self.params.copy()
        current_lr = learning_rate
        
        has_val = m_val is not None and n_val is not None and y_val is not None
        if has_val:
            m_val, n_val, y_val = np.array(m_val), np.array(n_val), np.array(y_val)
            y_curr, base_name = y_val, ""
        else:
            y_curr, base_name = t, "Train "
            
        metric_name = f"{base_name}Log Loss"
        
        for e in range(epochs):
            grads = self._compute_gradients(m_train, t, n_train, l2_reg)
            
            w = beta1 * w + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * (grads ** 2)
            
            w_hat = w / (1 - beta1 ** (e + 1))
            v_hat = v / (1 - beta2 ** (e + 1))
            
            self.params -= current_lr * w_hat / (np.sqrt(v_hat) + eps)
            
            pred_curr = self.predicted_value(m_val, n_val) if has_val else self.predicted_value(m_train, n_train)
            current_loss = log_loss(y_curr, pred_curr, labels=[0, 1])
            
            if current_loss < best_loss - 1e-7:
                best_loss = current_loss
                patience_counter = 0
                lr_patience_counter = 0  
                best_params = self.params.copy()
            else:
                patience_counter += 1
                lr_patience_counter += 1  
            
            if lr_patience_counter >= lr_patience:
                current_lr = max(current_lr * factor, min_lr)
                lr_patience_counter = 0  
                if verbose:
                    print(f"The model has plateaued. Lowering the learning rate to: {current_lr:.6f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {e+1}. Best {metric_name} = {best_loss:.6f}")
                self.params = best_params.copy()
                break
            
            if verbose and (e + 1) % 1000 == 0:
                print(f"Epoch {e+1}: {metric_name} = {current_loss:.6f} (LR: {current_lr:.6f})")

    def predict_proba(self, m_test, n_test):
        return self.predicted_value(np.array(m_test), np.array(n_test))
