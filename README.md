# AlgorithmTest — SHWFS Weighting Functions & Turbulence Modeling

## 📌 Overview

This notebook implements and compares **theoretical slope weighting functions** for Shack–Hartmann Wavefront Sensors (SHWFS) using both:

- Integral (Butterley) method  
- Fourier spectral method  
- Z-tilt and G-tilt formulations  
- Scintillation weighting functions  
- Tip/tilt subtraction


Primary reference:

> R. M. Griffiths, *Continuous 24-hour Shack-Hartmann optical turbulence profiling on a small telescope*, Durham University, 2024.

---

## 🎯 Objectives

- Implement Kolmogorov turbulence structure functions  
- Generate theoretical slope covariance maps  
- Implement Z-tilt spectral filters  
- Compute multi-layer Kolmogorov PSD  
- Derive weighting functions via Fourier methods  
- Compare Z-tilt vs G-tilt behavior  
- Apply tip/tilt subtraction  
- Diagnose scaling mismatches between methods  

---

## 📦 Dependencies

```bash
pip install numpy matplotlib aotools
```

## 👨‍🔬 Author Notes

- This notebook is part of ongoing research at OptoLab (PUCV) focused on:

- Event-based SHWFS

- Optical turbulence profiling

- SHIMM / SLODAR modeling

- Fourier vs integral covariance methods

## 📚 References

- Griffiths, R. M. (2024). Durham University Thesis
- Butterley theoretical SLODAR covariance code
- Kolmogorov turbulence theory