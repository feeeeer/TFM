"""
deepsurv_numpy.py
Implementacion propia de DeepSurv (Katzman et al., 2018) con NumPy: un perceptron
multicapa que predice el log-riesgo eta(x) y se entrena minimizando la
verosimilitud parcial de Cox negativa (aproximacion de Breslow para empates).
No usa PyTorch/TensorFlow: el entorno de trabajo no los tenia disponibles y, con
~7.000 filas y redes pequenas, NumPy en CPU es suficiente.

Uso:
    m = DeepSurv(hidden=(32, 16), l2=1e-3, lr=5e-3, epochs=300, seed=0)
    m.fit(X_train, duration, event)
    eta = m.predict_risk(X)            # mayor eta = mayor riesgo
    S = m.predict_survival(X, times)   # matriz (n, len(times))

Autor: Fernando Velasco Alonso
"""
import numpy as np


class DeepSurv:
    def __init__(self, hidden=(32, 16), l2=1e-3, lr=5e-3, epochs=300, seed=0, dropout=0.0):
        self.hidden, self.l2, self.lr, self.epochs = tuple(hidden), l2, lr, epochs
        self.seed, self.dropout = seed, dropout
        self.loss_history = []

    # ---------- verosimilitud parcial de Cox (Breslow) ----------
    @staticmethod
    def _tie_groups(t_sorted):
        _, first, counts = np.unique(t_sorted, return_index=True, return_counts=True)
        first_idx = np.repeat(first, counts)
        last_idx = np.repeat(first + counts - 1, counts)
        return first_idx, last_idx

    def _cox_loss_grad(self, eta, d, first_idx, last_idx):
        m = eta.max()
        e = np.exp(eta - m)
        S = np.cumsum(e[::-1])[::-1][first_idx]          # suma de riesgo del conjunto en riesgo
        n_ev = max(d.sum(), 1.0)
        loss = -np.sum(d * (eta - m - np.log(S))) / n_ev
        A = np.cumsum(d / S)[last_idx]                   # sum_{i evento, t_i <= t_k} 1/S_i
        grad = -(d - e * A) / n_ev
        return loss, grad

    # ---------- red ----------
    def _init(self, n_in):
        rng = np.random.default_rng(self.seed)
        sizes = (n_in,) + self.hidden + (1,)
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def _forward(self, X, train=False, rng=None):
        acts, masks, a = [X], [], X
        for i in range(len(self.W) - 1):
            a = np.maximum(a @ self.W[i] + self.b[i], 0.0)
            if train and self.dropout > 0:
                mask = (rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                a = a * mask
                masks.append(mask)
            else:
                masks.append(None)
            acts.append(a)
        out = (a @ self.W[-1] + self.b[-1]).ravel()
        return out, acts, masks

    def fit(self, X, duration, event):
        X = np.asarray(X, dtype=float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        order = np.argsort(duration, kind="stable")
        Xs, t, d = Xs[order], np.asarray(duration, float)[order], np.asarray(event, float)[order]
        first_idx, last_idx = self._tie_groups(t)
        self._init(Xs.shape[1])
        rng = np.random.default_rng(self.seed + 1)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.loss_history = []
        for step in range(1, self.epochs + 1):
            eta, acts, masks = self._forward(Xs, train=True, rng=rng)
            loss, g = self._cox_loss_grad(eta, d, first_idx, last_idx)
            reg = 0.5 * self.l2 * sum((w ** 2).sum() for w in self.W)
            self.loss_history.append(loss + reg)
            delta = g[:, None]
            for i in range(len(self.W) - 1, -1, -1):
                gW = acts[i].T @ delta + self.l2 * self.W[i]
                gb = delta.sum(0)
                if i > 0:
                    delta = delta @ self.W[i].T
                    if masks[i - 1] is not None:
                        delta = delta * masks[i - 1]
                    delta = delta * (acts[i] > 0)
                mW[i] = b1 * mW[i] + (1 - b1) * gW; vW[i] = b2 * vW[i] + (1 - b2) * gW ** 2
                mb[i] = b1 * mb[i] + (1 - b1) * gb; vb[i] = b2 * vb[i] + (1 - b2) * gb ** 2
                self.W[i] -= self.lr * (mW[i] / (1 - b1 ** step)) / (np.sqrt(vW[i] / (1 - b2 ** step)) + eps)
                self.b[i] -= self.lr * (mb[i] / (1 - b1 ** step)) / (np.sqrt(vb[i] / (1 - b2 ** step)) + eps)
        # riesgo base de Breslow con los datos de entrenamiento
        eta_tr, _, _ = self._forward(Xs)
        e = np.exp(eta_tr - eta_tr.max())
        S = np.cumsum(e[::-1])[::-1][first_idx] * np.exp(eta_tr.max())
        ut, idx = np.unique(t, return_index=True)
        dh = np.array([d[(t == u)].sum() / S[i] for u, i in zip(ut, idx)])
        self._ut, self._H0 = ut, np.cumsum(dh)
        return self

    def predict_risk(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return self._forward(Xs)[0]

    def predict_survival(self, X, times):
        eta = self.predict_risk(X)
        pos = np.searchsorted(self._ut, np.asarray(times, float), side="right") - 1
        H0 = np.where(pos >= 0, self._H0[np.clip(pos, 0, None)], 0.0)
        return np.exp(-np.outer(np.exp(eta), H0))


def _self_check():
    """Comprueba el gradiente analitico contra diferencias finitas y que aprende senal."""
    rng = np.random.default_rng(0)
    n = 60
    t = rng.integers(1, 15, n).astype(float)
    d = (rng.random(n) < 0.6).astype(float)
    o = np.argsort(t, kind="stable"); t, d = t[o], d[o]
    f, l = DeepSurv._tie_groups(t)
    eta = rng.normal(size=n)
    m = DeepSurv()
    _, g = m._cox_loss_grad(eta, d, f, l)
    k, h = 7, 1e-6
    e1, e2 = eta.copy(), eta.copy(); e1[k] += h; e2[k] -= h
    num = (m._cox_loss_grad(e1, d, f, l)[0] - m._cox_loss_grad(e2, d, f, l)[0]) / (2 * h)
    assert abs(num - g[k]) < 1e-6, (num, g[k])
    X = rng.normal(size=(400, 3)); tt = rng.exponential(np.exp(-X[:, 0])) * 10 + 1
    ev = (rng.random(400) < 0.8).astype(float)
    ds = DeepSurv(hidden=(8,), epochs=150).fit(X, np.ceil(tt), ev)
    from lifelines.utils import concordance_index
    assert concordance_index(tt, -ds.predict_risk(X), ev) > 0.65
    print("self-check DeepSurv OK")


if __name__ == "__main__":
    _self_check()
