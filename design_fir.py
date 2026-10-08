"""
Channel-select FIR for 4.8 kbps GMSK in a 7.5 kHz channel.
fs = 90 kHz, 125 taps (Type I linear phase, group delay = 62 samples = 0.689 ms).
Applied identically to I and Q at complex baseband.

Evaluates each candidate against:
  * A-M3 adjacent interferer: FM, 400 Hz tone, 900 Hz deviation (12 % of 7.5 kHz), +7.5 kHz offset
  * Self-distortion of the wanted GMSK (NMSE vs. delayed unfiltered signal)
"""
import numpy as np
from scipy import signal
from scipy.special import ndtr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 90_000.0
NTAPS = 125
RB = 4_800.0
T = 1 / RB
F_OFF = 7_500.0            # adjacent channel offset
F_MOD = 400.0              # A-M3 audio tone
DEV = 0.12 * 7_500.0       # 900 Hz deviation
ACS_DB = 60.0              # interferer-to-wanted ratio at the antenna for the test
rng = np.random.default_rng(1)


def gmsk(nsym, bt):
    """Complex-baseband GMSK, h = 0.5, sampled at FS (18.75 samples/symbol)."""
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt * RB)
    bits = rng.integers(0, 2, nsym) * 2 - 1
    n = int(nsym * T * FS)
    t = np.arange(n) / FS
    freq = np.zeros(n)
    for k, a in enumerate(bits):
        tc = (k + 0.5) * T
        lo, hi = int(max(0, (tc - 4 * T) * FS)), int(min(n, (tc + 4 * T) * FS))
        tt = t[lo:hi] - tc
        freq[lo:hi] += a * (ndtr((tt + T / 2) / sigma) - ndtr((tt - T / 2) / sigma))
    phase = 2 * np.pi * np.cumsum(freq * RB / 4) / FS     # peak deviation Rb/4 = 1.2 kHz
    return np.exp(1j * phase)


def am3(n):
    t = np.arange(n) / FS
    beta = DEV / F_MOD
    return np.exp(1j * (2 * np.pi * F_OFF * t + beta * np.sin(2 * np.pi * F_MOD * t)))


def evaluate(h, wanted, interf):
    d = (len(h) - 1) // 2
    w_out = signal.lfilter(h, 1, wanted)[2 * d:]
    i_out = signal.lfilter(h, 1, interf)[2 * d:]
    ref = wanted[d:len(wanted) - d]
    g = np.vdot(ref, w_out) / np.vdot(ref, ref)        # best complex gain
    nmse = 10 * np.log10(np.mean(np.abs(w_out - g * ref) ** 2) / np.mean(np.abs(g * ref) ** 2))
    rej = 10 * np.log10(np.mean(np.abs(interf) ** 2) / np.mean(np.abs(i_out) ** 2)) \
        - 10 * np.log10(np.mean(np.abs(wanted) ** 2) / np.mean(np.abs(w_out) ** 2))
    return rej, nmse


def remez_lp(fp, fst, wstop):
    return signal.remez(NTAPS, [0, fp, fst, FS / 2], [1, 0], weight=[1, wstop], fs=FS)


def mag_db(h, f):
    _, H = signal.freqz(h, worN=np.asarray(f), fs=FS)
    return 20 * np.log10(np.abs(H) + 1e-15)


if __name__ == "__main__":
    results = {}
    for bt in (0.3, 0.5):
        wanted = gmsk(3000, bt)
        interf = am3(len(wanted))
        best = None
        rows = []
        for fp in np.arange(1800, 3001, 100):
            for fst in np.arange(4000, 6001, 100):
                for wst in (1, 3, 10, 30):
                    h = remez_lp(fp, fst, wst)
                    rej, nmse = evaluate(h, wanted, interf)
                    rows.append((fp, fst, wst, rej, nmse))
        rows = np.array(rows)
        # Pick max interferer rejection subject to wanted-signal distortion <= -30 dB NMSE
        ok = rows[rows[:, 4] <= -30]
        pick = ok[np.argmax(ok[:, 3])]
        results[bt] = (pick, rows, wanted, interf)
        print(f"BT={bt}: best fp={pick[0]:.0f} Hz fst={pick[1]:.0f} Hz Wstop={pick[2]:.0f} "
              f"-> ACS rejection {pick[3]:.1f} dB, wanted NMSE {pick[4]:.1f} dB, "
              f"post-filter C/I at +60 dB test = {pick[3] - ACS_DB:+.1f} dB")

    # Final design uses the BT=0.3 pick (worst case is the wider BT=0.5 spectrum; report both)
    pick, rows, wanted, interf = results[0.3]
    fp, fst, wst = pick[0], pick[1], pick[2]
    h = remez_lp(fp, fst, wst)
    rej, nmse = evaluate(h, wanted, interf)
    rej05, nmse05 = evaluate(h, results[0.5][2], results[0.5][3])

    # 16-bit quantisation check
    q = np.round(h / np.max(np.abs(h)) * 32767) / 32767 * np.max(np.abs(h))
    q15 = np.round(h * 32768).astype(int)   # Q1.15 (all |h| < 1)
    rej_q, nmse_q = evaluate(q15 / 32768, wanted, interf)

    pts = [2400, 3000, 4000, 5100, 5500, 5900, 6300, 6700, 7100, 7500, 8000, 10000, 20000]
    att = -mag_db(h, pts)
    att_q = -mag_db(q15 / 32768, pts)
    f_full = np.linspace(0, FS / 2, 8192)
    m = mag_db(h, f_full)
    ripple = np.ptp(m[f_full <= fp])
    stop_min = -np.max(m[f_full >= fst])

    with open("design_report.txt", "w") as fh:
        fh.write(f"fs={FS:.0f} Hz, taps={NTAPS}, group delay={(NTAPS-1)//2} samples = {(NTAPS-1)/2/FS*1e3:.3f} ms\n")
        fh.write(f"Parks-McClellan: passband 0-{fp:.0f} Hz, stopband {fst:.0f}-45000 Hz, stop weight {wst:.0f}\n")
        fh.write(f"Passband ripple {ripple:.3f} dB p-p, min stopband attenuation {stop_min:.1f} dB\n")
        fh.write(f"Sum of taps (DC gain) {np.sum(h):.6f}\n\n")
        fh.write(f"A-M3 rejection (BT 0.3 wanted): {rej:.1f} dB  -> C/I after filter at +60 dB = {rej-ACS_DB:+.1f} dB\n")
        fh.write(f"A-M3 rejection (BT 0.5 wanted): {rej05:.1f} dB  -> C/I after filter at +60 dB = {rej05-ACS_DB:+.1f} dB\n")
        fh.write(f"Wanted-signal NMSE (filter distortion): BT0.3 {nmse:.1f} dB, BT0.5 {nmse05:.1f} dB\n")
        fh.write(f"Q1.15 coefficients: rejection {rej_q:.1f} dB, NMSE {nmse_q:.1f} dB\n\n")
        fh.write("Offset(Hz)  Atten float(dB)  Atten Q1.15(dB)\n")
        for f, a, b in zip(pts, att, att_q):
            fh.write(f"{f:9d}  {a:14.1f}  {b:14.1f}\n")
    print(open("design_report.txt").read())

    np.savetxt("fir_coeffs_float.txt", h, fmt="%.12e")
    with open("fir_coeffs_q15.h", "w") as fh:
        fh.write(f"/* GMSK 4.8 kbps channel filter, fs=90 kHz, {NTAPS} taps, Q1.15, symmetric */\n")
        fh.write(f"/* pass 0-{fp:.0f} Hz, stop >= {fst:.0f} Hz, apply to I and Q */\n")
        fh.write(f"#define GMSK_CHFIR_NTAPS {NTAPS}\n")
        fh.write("static const short gmsk_chfir_q15[GMSK_CHFIR_NTAPS] = {\n")
        for i in range(0, NTAPS, 8):
            fh.write("    " + ", ".join(f"{v:6d}" for v in q15[i:i + 8]) + ",\n")
        fh.write("};\n")

    # Plots
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    ax[0].plot(f_full / 1e3, m, label="FIR (float)")
    ax[0].plot(f_full / 1e3, mag_db(q15 / 32768, f_full), "--", lw=0.8, label="FIR (Q1.15)")
    beta = DEV / F_MOD
    from scipy.special import jv
    for k in range(-8, 9):
        f = F_OFF + k * F_MOD
        lvl = 20 * np.log10(abs(jv(k, beta)) + 1e-12)
        need = -(ACS_DB + 10 + lvl + 3)        # C/I 10 dB + 3 dB margin
        ax[0].plot(f / 1e3, need, "rv" if need < 0 else "w", ms=6)
    ax[0].plot([], [], "rv", label="required atten. per A-M3 line (C/I 10 dB + 3 dB)")
    ax[0].axvline(fp / 1e3, color="g", ls=":", lw=0.8)
    ax[0].axvline(fst / 1e3, color="g", ls=":", lw=0.8)
    ax[0].set(xlim=(0, 20), ylim=(-110, 5), xlabel="Frequency offset (kHz)", ylabel="dB",
              title=f"{NTAPS}-tap channel FIR @ 90 kHz  (pass {fp/1e3:.1f} kHz / stop {fst/1e3:.1f} kHz)")
    ax[0].grid(alpha=0.3); ax[0].legend(loc="upper right", fontsize=8)

    def psd(x):
        # Averaged, windowed periodogram (50 % overlap), steady-state samples only
        nfft, win = 4096, signal.windows.blackmanharris(4096)
        x = x[NTAPS:]
        segs = [x[i:i + nfft] * win for i in range(0, len(x) - nfft, nfft // 2)]
        p = np.mean([np.abs(np.fft.fft(s)) ** 2 for s in segs], axis=0) / (FS * np.sum(win ** 2))
        f = np.fft.fftfreq(nfft, 1 / FS)
        return np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(p) + 1e-30)
    scale = 10 ** (ACS_DB / 20)
    f, pw = psd(wanted); _, pi = psd(scale * interf)
    _, pwf = psd(signal.lfilter(h, 1, wanted)); _, pif = psd(signal.lfilter(h, 1, scale * interf))
    ax[1].plot(f / 1e3, pw, color="0.6", label="wanted GMSK in")
    ax[1].plot(f / 1e3, pi, color="orange", alpha=0.6, label="A-M3 in (+60 dB)")
    ax[1].plot(f / 1e3, pwf, "b", label="wanted after FIR")
    ax[1].plot(f / 1e3, pif, "r", label="A-M3 after FIR")
    ax[1].set(xlim=(-15, 15), xlabel="Frequency (kHz)", ylabel="dB/Hz (rel.)",
              title=f"ACS test: post-filter C/I = {rej - ACS_DB:+.1f} dB")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fir_response.png", dpi=110)
