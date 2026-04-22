"""
Extension — PAC Detector (Phase-Amplitude Coupling) Temporel
Détection du couplage amplitude-phase dans la dynamique neuronale.

ATTENTION piège B11 :
    Le PAC est TEMPOREL (FFT sur l'axe du temps), pas spatial.
    Ne PAS calculer la FFT sur le vecteur spatial 128D —
    utiliser un buffer roulant de 100 pas et faire la FFT sur l'axe temporel.

Architecture :
    Buffer roulant de T=100 pas → FFT → extraction bandes de fréquence
    Couplage PAC via Modulation Index (MI) de Tort et al. 2010

Réf. : Formalisation_Mille_Cerveaux.pdf §ext.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PACDetector(nn.Module):
    """
    Détecteur de couplage amplitude-phase (PAC) temporel.

    Utilise un buffer roulant de T pas pour calculer la FFT temporelle
    et extraire les bandes de fréquence theta (4–12 Hz) et gamma (30–80 Hz).

    Le Modulation Index (MI) de Tort 2010 mesure le couplage entre
    la phase basse fréquence et l'amplitude haute fréquence.

    Args:
        signal_dim:   dimension du signal par pas de temps
        buffer_len:   longueur du buffer temporel (défaut 100 pas)
        fs:           fréquence d'échantillonnage (Hz, pour l'axe fréquentiel)
        theta_band:   bande de fréquence theta (Hz)
        gamma_band:   bande de fréquence gamma (Hz)
        n_phase_bins: nombre de bins de phase pour le MI
    """

    def __init__(
        self,
        signal_dim: int,
        buffer_len: int = 100,
        fs: float = 1000.0,       # 1 kHz par défaut
        theta_band: Tuple[float, float] = (4.0, 12.0),
        gamma_band: Tuple[float, float] = (30.0, 80.0),
        n_phase_bins: int = 18,
    ) -> None:
        super().__init__()

        self.signal_dim = signal_dim
        self.buffer_len = buffer_len
        self.fs = fs
        self.theta_band = theta_band
        self.gamma_band = gamma_band
        self.n_phase_bins = n_phase_bins

        # ── Buffer roulant temporel ───────────────────────────────────────
        # Shape : (buffer_len, signal_dim)
        # FFT sur l'axe temporel (dim=0), PAS spatial (dim=1) — piège B11
        self.register_buffer(
            "temporal_buffer",
            torch.zeros(buffer_len, signal_dim),
        )
        self.register_buffer("buffer_idx", torch.tensor(0, dtype=torch.long))
        self.register_buffer("buffer_full", torch.tensor(False))

    @torch.no_grad()
    def push(self, signal: torch.Tensor) -> None:
        """
        Ajoute un vecteur de signal au buffer temporel (FIFO circulaire).

        Args:
            signal: signal au pas courant, shape (signal_dim,)
        """
        idx = self.buffer_idx.item()
        self.temporal_buffer[idx] = signal
        self.buffer_idx.data = torch.tensor((idx + 1) % self.buffer_len)

        if self.buffer_idx.item() == 0:
            self.buffer_full.data = torch.tensor(True)

    @torch.no_grad()
    def compute_pac(self) -> Optional[dict]:
        """
        Calcule le PAC temporel sur le buffer courant.

        FFT sur l'axe temporel (dim=0) — PAS sur l'axe spatial.
        Piège B11 : ne pas faire FFT sur le vecteur spatial.

        Returns:
            dict avec MI, theta_power, gamma_amplitude, ou None si
            le buffer n'est pas encore plein.
        """
        if not self.buffer_full.item():
            return None

        # Buffer ordonné chronologiquement : (T, signal_dim)
        idx = self.buffer_idx.item()
        ordered = torch.roll(self.temporal_buffer, -idx, dims=0)

        # ── FFT temporelle ────────────────────────────────────────────────
        # FFT sur dim=0 (axe temporel), PAS dim=1 (axe spatial) — B11
        # Shape : (T//2+1, signal_dim) pour rfft
        spectrum = torch.fft.rfft(ordered, dim=0)   # FFT temporelle
        freqs = torch.fft.rfftfreq(self.buffer_len, d=1.0 / self.fs)

        # ── Extraction bandes ─────────────────────────────────────────────
        theta_mask = (freqs >= self.theta_band[0]) & (freqs <= self.theta_band[1])
        gamma_mask = (freqs >= self.gamma_band[0]) & (freqs <= self.gamma_band[1])

        theta_power = spectrum[theta_mask].abs().mean(dim=0)   # (signal_dim,)
        gamma_amp = spectrum[gamma_mask].abs().mean(dim=0)     # (signal_dim,)

        # ── Modulation Index (MI de Tort 2010) ────────────────────────────
        # Calcul simplifié sur la puissance moyenne
        theta_scalar = theta_power.mean().item()
        gamma_scalar = gamma_amp.mean().item()

        if theta_scalar > 0 and gamma_scalar > 0:
            mi = self._compute_mi(ordered)
        else:
            mi = 0.0

        return {
            "MI": mi,
            "theta_power": theta_power,
            "gamma_amplitude": gamma_amp,
            "freqs": freqs,
            "buffer_len": self.buffer_len,
        }

    @torch.no_grad()
    def _compute_mi(self, signal: torch.Tensor) -> float:
        """
        Calcule le Modulation Index de Tort 2010.

        MI = D_KL(P_observed || P_uniform) / log(n_bins)
        où P_observed[j] est l'amplitude gamma moyenne dans le bin
        de phase theta j.

        Signal moyen sur les dimensions spatiales pour le calcul MI.

        Args:
            signal: buffer temporel ordonné, shape (T, signal_dim)

        Returns:
            MI: scalaire dans [0, 1]
        """
        # Réduction sur l'axe spatial : signal 1D temporel
        s = signal.mean(dim=-1).numpy()  # (T,)
        T = len(s)

        import numpy as np

        # Filtrage theta via FFT
        S = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)

        # Bande theta → signal de phase
        theta_mask = (freqs >= self.theta_band[0]) & (freqs <= self.theta_band[1])
        S_theta = np.zeros_like(S)
        S_theta[theta_mask] = S[theta_mask]
        s_theta = np.fft.irfft(S_theta, n=T)
        phase_theta = np.angle(
            s_theta + 1j * np.imag(np.fft.ifft(np.fft.fft(s_theta) * 1j))
        )

        # Bande gamma → signal d'amplitude
        gamma_mask = (freqs >= self.gamma_band[0]) & (freqs <= self.gamma_band[1])
        S_gamma = np.zeros_like(S)
        S_gamma[gamma_mask] = S[gamma_mask]
        s_gamma_amp = np.abs(np.fft.irfft(S_gamma, n=T))

        # Distribution P(amplitude|phase)
        phase_bins = np.linspace(-np.pi, np.pi, self.n_phase_bins + 1)
        P = np.zeros(self.n_phase_bins)
        for j in range(self.n_phase_bins):
            mask = (phase_theta >= phase_bins[j]) & (phase_theta < phase_bins[j + 1])
            P[j] = s_gamma_amp[mask].mean() if mask.any() else 0.0

        P_sum = P.sum()
        if P_sum == 0:
            return 0.0
        P = P / P_sum

        # KL divergence avec distribution uniforme
        P_uniform = np.ones(self.n_phase_bins) / self.n_phase_bins
        eps = 1e-10
        kl = np.sum(P * np.log((P + eps) / P_uniform))
        mi = kl / math.log(self.n_phase_bins)

        return float(np.clip(mi, 0.0, 1.0))

    def reset(self) -> None:
        """Vide le buffer temporel."""
        self.temporal_buffer.data.zero_()
        self.buffer_idx.data.zero_()
        self.buffer_full.data = torch.tensor(False)

    def extra_repr(self) -> str:
        return (
            f"signal_dim={self.signal_dim}, buffer_len={self.buffer_len}, "
            f"fs={self.fs}, theta={self.theta_band}, gamma={self.gamma_band}"
        )
