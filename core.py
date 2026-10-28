import numpy as np

W = 2 * np.pi / 24          # angular frequency (rad/hour)

# ---------- Model, loss, gradients (calculus) ----------
def energy(t, A, phi, M):
    return A * np.sin(W * (t - phi)) + M

def loss(t, y, A, phi, M):
    return np.mean((energy(t, A, phi, M) - y) ** 2)

def gradients(t, y, A, phi, M):
    s = np.sin(W * (t - phi))
    c = np.cos(W * (t - phi))
    err = A * s + M - y
    dA = 2 * np.mean(err * s)
    dphi = -2 * A * W * np.mean(err * c)     # chain rule
    dM = 2 * np.mean(err)
    return dA, dphi, dM

# ---------- Gradient descent ----------
def init_guess(t, y, A0=2.0):
    """Sine curves have many local minima in phi, so pick a smart starting phase."""
    M0 = np.mean(y)
    best_phi = min(np.arange(0, 24, 1.0), key=lambda p: loss(t, y, A0, p, M0))
    return A0, best_phi, M0

def fit(t, y, lr_A=0.05, lr_phi=0.5, lr_M=0.05, epochs=5000, init=None):
    A, phi, M = init if init else init_guess(t, y)
    history = []
    for _ in range(epochs):
        dA, dphi, dM = gradients(t, y, A, phi, M)
        A -= lr_A * dA
        phi -= lr_phi * dphi
        M -= lr_M * dM
        history.append(loss(t, y, A, phi, M))
    if A < 0:                       # same curve, shifted half a day
        A, phi = -A, phi + 12
    return A, phi % 24, M, history

# ---------- Insights ----------
def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)

def peak_time(phi):
    return (phi + 6) % 24           # dE/dt = 0, maximum

def daytime_low(A, phi, M, lo=7.0, hi=23.0):
    """Lowest point while awake. The true minimum (phi+18) may fall at night,
    so we also check the interval endpoints."""
    grid = np.linspace(lo, hi, 400)
    return grid[np.argmin(energy(grid, A, phi, M))]

def window_integral(a, b, A, phi, M):
    """Exact integral of E(t) from a to b."""
    return M * (b - a) - (A / W) * (np.cos(W * (b - phi)) - np.cos(W * (a - phi)))

def best_window(A, phi, M, length=3.0, lo=6.0, hi=23.0):
    starts = np.arange(lo, hi - length + 0.01, 0.25)
    vals = np.array([window_integral(s, s + length, A, phi, M) for s in starts])
    i = np.argmax(vals)
    return starts[i], starts[i] + length, vals[i] / length   # avg energy in window

def sleep_debt(sleep_hours, need=8.0):
    debt = 0.0
    for s in sleep_hours:
        debt = max(0.0, debt + (need - s))
    return debt

def fmt_hour(h):
    h = h % 24
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    hh %= 24
    suffix = "AM" if hh < 12 else "PM"
    return f"{(hh % 12) or 12}:{mm:02d} {suffix}"
