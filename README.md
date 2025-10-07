# 🧠 DLSH — CUDA-Accelerated Locality Sensitive Hashing

**Authors:** [Zakaria El Kazdam](https://github.com/ZakariaElKazdam) & [Riham Faraj](https://github.com/farajriham)  
**Date:** February 2025  

---

## 📘 Overview

This project implements a **Locality-Sensitive Hashing (LSH)** system for **approximate nearest neighbor search** in high-dimensional spaces.

It includes:
- A **sequential CPU version** to validate correctness and benchmark performance.
- A **parallel CUDA version** leveraging GPU acceleration for massive speedups.

> 💡 For detailed technical explanations, please refer to the [Project Report](./project_report.pdf).

---

## 🚀 Motivation

As datasets grow in size and dimensionality, sequential LSH becomes computationally expensive.  
This project explores how **GPU parallelism** (via CUDA) can:
- Greatly **reduce computation time**,  
- Optimize **memory access and data transfer**,  
- Ensure **scalability** for large-scale similarity search problems.

---

## 🧩 Project Structure

| Folder/File | Description |
|--------------|-------------|
| `DLSH_algo.cpp` / `.h` | Core LSH algorithm: hash table creation and neighbor search. |
| `hashing.cpp` / `.h` | Mathematical hash operations and random projection functions. |
| `NewPoint.cpp` / `.h` | Operations for new data points and lookup mechanisms. |
| `make_csv.cpp` | Converts NumPy (`.npy`) data files to CSV for preprocessing. |
| `CUDA/` | Contains GPU kernels and CUDA functions (`generateLSHParams`, `computeHashes`, etc.). |
| `Data/` | Sample dataset and output CSVs. |
| `project_report.pdf` | Full technical documentation (architecture, performance, CUDA breakdown). |

---

## 🧮 Sequential Implementation

- Implements standard LSH algorithm for nearest neighbor search.  
- Uses random hyperplanes to hash points into buckets.  
- Employs C++ STL containers (`std::map`, `std::set`) for efficient storage.  

**Key Concepts:**
- Hash function:  
  \[
  h(x) = \frac{a \cdot x + b}{w}
  \]
- Multi-level hashing for better accuracy (`L1`, `L2` layers).  
- Complexity:  
  - Hash computation: O(d)  
  - Lookup: Sublinear time  

---

## ⚙️ Parallel CUDA Implementation

### 🔹 CUDA Kernels
- **`generateLSHParams`**  
  Initializes random vectors and biases (`a`, `b`) for hash functions in parallel.
- **`computeHashes`**  
  Computes hash values for each data point using thousands of GPU threads.
- **`hashingComputingCUDA`**  
  Performs dot products and hash computation inside each thread for one data point.

### 🔹 Memory & Execution
- Custom memory management using raw pointers (`double*`) to GPU memory.  
- Avoids unnecessary data transfers between CPU and GPU.  
- Achieves significant speedups for large datasets.

---

## 📊 Results & Observations

- GPU implementation achieves **massive performance gains** over the sequential version.  
- The final system is both **scalable** and **memory-efficient**.  
- A structured pipeline from data loading → hashing → bucket assignment → retrieval.

> 🧾 See *Section 3: Parallel Implementation* in the [Project Report](./project_report.pdf) for in-depth kernel explanations and figures.

---

## 🛠️ Future Work

- Parallelize the dot product computation (`hashingComputingCUDA`) using **warp-level reductions**.  
- Integrate **CUDA memory debugging tools** (`cuda-memcheck`).  
- Add adaptive parameter tuning for `w`, `L1`, `L2`.  
- Implement **benchmarking scripts** comparing CPU vs GPU performance.

---

## 🧰 Build & Run

### 🖥️ Requirements
- CUDA Toolkit ≥ 11.0  
- C++17 compatible compiler (g++)  
- NVIDIA GPU  

### ⚙️ Compilation Example
```bash
nvcc -o DLSH main.cu hashing.cu computeHashes.cu -std=c++17
./DLSH
```

## 🧑‍💻 Authors

- **Zakaria El Kazdam** — GPU kernel design, CUDA integration, report writing  
- **Riham Faraj** — Sequential implementation, data preprocessing, testing  

---

## 📄 Reference

For the full implementation details, explanations, and design diagrams,  
please check the complete [📘 Project Report (PDF)](./project_report.pdf).