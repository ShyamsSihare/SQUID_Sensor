import numpy as np
import physics as ph

FREQ = ph.FREQ_BINS  # 256 log-spaced bins, 1e-2..1e4 Hz


def _jitter(rng, shape, sigma_db=0.6):
    return np.exp(rng.normal(0, sigma_db * np.log(10) / 10, size=shape))


def healthy_spectrum(rng, t_seconds=None):
    if t_seconds is None:
        t_seconds = rng.uniform(0, 1.0) * ph.SECONDS_PER_YEAR
    S = ph.total_noise_spectrum(FREQ, t_seconds)
    return S * _jitter(rng, FREQ.shape)


def anomaly_spectrum(rng, mode):
    if mode == "sensor_degradation":
        # advanced, but not yet catastrophic, accumulated damage
        t = rng.uniform(5.0, 7.0) * ph.SECONDS_PER_YEAR
        S = ph.total_noise_spectrum(FREQ, t)

    elif mode == "radiation_burst":
        # otherwise mid-life sensor hit by a transient flux spike
        t = rng.uniform(1.0, 4.0) * ph.SECONDS_PER_YEAR
        burst = rng.uniform(15, 40)
        S = ph.total_noise_spectrum(FREQ, t, burst_factor=burst)

    elif mode == "coolant_anomaly":
        # loss of cooling flow -> elevated bath temperature raises the
        # thermal (Tesche-Clarke) term, eq:thermal_noise
        t = rng.uniform(0, 2.0) * ph.SECONDS_PER_YEAR
        T_elevated = ph.T_op * rng.uniform(2.5, 5.0)
        S = ph.total_noise_spectrum(FREQ, t, T=T_elevated)

    elif mode == "foreign_material":
        # baseline moderate-life spectrum + a localized resonance bump
        # from flow-induced vibration around a mid-band frequency
        t = rng.uniform(1.0, 4.0) * ph.SECONDS_PER_YEAR
        S = ph.total_noise_spectrum(FREQ, t)
        f0 = 10 ** rng.uniform(1.0, 2.5)  # resonance center, 10-300 Hz
        width = f0 * 0.15
        bump = rng.uniform(3, 10) * S.max() * np.exp(-0.5 * ((FREQ - f0) / width) ** 2)
        S = S + bump

    elif mode == "calibration_drift":
        # otherwise healthy sensor, systematic multiplicative gain error
        t = rng.uniform(0, 2.0) * ph.SECONDS_PER_YEAR
        S = ph.total_noise_spectrum(FREQ, t)
        drift = rng.uniform(2.5, 6.0)
        S = S * drift

    else:
        raise ValueError(mode)

    return S * _jitter(rng, FREQ.shape)


FAILURE_MODES = [
    "sensor_degradation", "radiation_burst", "coolant_anomaly",
    "foreign_material", "calibration_drift",
]
