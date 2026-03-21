# DCAE-Based Data Assimilation for the Kuroshio Extension

## Background

This project applies a **Deep Convolutional AutoEncoder (DCAE)** to ocean data assimilation in the Kuroshio Extension region. The input data consists of high-dimensional, multi-channel 3D ocean state variables with **101 vertical levels**, including physical fields such as temperature and salinity.

## Problem Statement

During experiments, we identified that DCAE exhibits a **sign flip phenomenon** in the latent space when encoding and decoding high-dimensional multi-channel data. Specifically, certain latent variables undergo non-physical sign reversals during the assimilation iteration, leading to spurious structures in the reconstructed physical fields after decoding.

Further analysis reveals that **this phenomenon is highly sensitive to initial conditions**: different background states or ensemble members trigger sign flips of varying severity, suggesting that DCAE's latent space suffers from discontinuities or rotational invariance issues.

## Comparison with VQ-VAE

| Property | DCAE | VQ-VAE |
|---|---|---|
| Compression Ratio | High ✅ | Low ❌ |
| Reconstruction Error | Low ✅ | High ❌ |
| Assimilation Stability | Unstable ❌ | Stable ✅ |

**VQ-VAE (Vector Quantized Variational AutoEncoder)** avoids continuous drift of latent variables through its discrete codebook mechanism, resulting in significantly greater stability during assimilation. However, this comes at the cost of a lower compression ratio and higher reconstruction error.

In contrast, DCAE offers a higher compression ratio and lower reconstruction error, making it theoretically better suited as a reduced-order surrogate space for data assimilation.

## Objective

The core goal of this project is to **resolve the instability of DCAE in multi-channel ocean data assimilation**, while preserving its advantages of high compression ratio and low reconstruction error.

Specifically, the project aims to:

- Investigate the root cause of the sign flip phenomenon in DCAE's latent space
- Develop methods to regularize or constrain the latent space to prevent non-physical sign reversals
- Achieve assimilation stability comparable to VQ-VAE without sacrificing DCAE's compression and reconstruction performance

## Repository Structure

> *(To be completed)*

## Getting Started

> *(To be completed)*

## References

> *(To be completed)*
