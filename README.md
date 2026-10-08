# GMSK Channel FIR Design

Channel-select FIR filter design for a 4.8 kbps GMSK receiver in a 7.5 kHz channel (fs = 90 kHz, 125 taps), with checks against the ETSI A-M3 adjacent-channel interferer. Includes a browser-based FIR filter designer.

## Contents

| File | What it is |
|---|---|
| `fir_designer.html` | Standalone FIR filter designer. Open it in any browser; no install or internet needed. |
| `design_fir.py` | Python script that designs the 125-tap filter and simulates the A-M3 adjacent-channel test |
| `cmx983_coeffs.txt` | The CMX983 FIR coefficients currently in use (16-bit integers, sum ≈ 2^18) |
| `fir_coeffs_q15.h` | Q1.15 coefficients from `design_fir.py` as a C array |
| `fir_coeffs_float.txt` | Floating-point coefficients from `design_fir.py` |
| `design_report.txt` | Measurements of the `design_fir.py` filter |
| `fir_response.png` | Frequency response and simulated ACS spectrum |

## FIR filter designer (`fir_designer.html`)

- **Response types:** lowpass, highpass, bandpass, bandstop
- **Methods:** equiripple (Parks-McClellan), least-squares, window (Kaiser, Hamming, Hann, Blackman, Blackman-Harris, rectangular)
- **Order:** set the number of taps, or find the minimum order for given Apass / Astop targets
- **Fixed point:** coefficient width, plus three scalings: passband gain = 2^S (CMX983 style), Q1.(bits−1), or peak at full scale
- **Plots:** magnitude, phase, group delay and impulse response, with hover readout and pinned data tips
- **Compare:** overlay any existing coefficient set (the CMX983 filter is preloaded)
- **FM interferer check:** required vs achieved attenuation for each FM sideband, and the ACS limit set by the filter
- **Export:** integer list, C array, CSV, floating point, or a JSON design record

## Python design script

```bash
pip install numpy scipy matplotlib
python design_fir.py
```

The script writes `design_report.txt`, `fir_coeffs_float.txt`, `fir_coeffs_q15.h` and `fir_response.png`.

## Key results

| Filter | A-M3 rejection | Filter-limited ACS (C/I 8–13 dB) |
|---|---|---|
| CMX983 coefficients (16-bit, −3 dB at 3.24 kHz, 85 dB stopband) | 88.3 dB | 75–80 dB |
| `design_fir.py`, Q1.15 coefficients | 84.5 dB | 71–76 dB |
| `design_fir.py`, floating point | 96.9 dB | 84–89 dB |

The filter alone exceeds the 60 dB ACS requirement. In practice, receiver LO phase noise at 7.5 kHz offset (aim for −105 to −110 dBc/Hz or better) and the test generator's own noise are the likely limits.
