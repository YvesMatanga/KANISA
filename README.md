# KANISA: Computational Library for Optimisation, Numerical Analysis, Dynamic Systems, Machine Learning, and Decision Analytics

**KANISA** (meaning *“think”*) is a unified computational framework designed to support advanced research and applied development in:

- Global and local optimisation  
- Numerical analysis  
- Dynamic systems and control  
- Machine learning  
- Data-driven decision analytics  

KANISA aims to provide a rigorous, extensible, and transparent environment for algorithmic development, benchmarking, and reproducible scientific research.


---

## How to Cite KANISA

If you use KANISA in your research, please follow the citation guidelines below:

➡️ **[View the Complete Citation File](./citation.md)**

---

## Library Structure

KANISA is organised into two primary components:

---

## 1. Python Library (Core Component)

The main KANISA library is implemented in Python and provides modules for:

- Global optimisation algorithms  
- Local search and numerical optimisation routines  
- Interval and uncertainty methods (planned)  
- Dynamic system modelling and control  
- Machine learning utilities  
- Decision intelligence and multi-criteria frameworks  

This Python library is under active development and will evolve into the primary, fully supported KANISA environment.

---

## 2. MATLAB Legacy Implementations

KANISA includes a dedicated `matlab/` directory preserving the original MATLAB implementations of key optimisation algorithms developed during earlier academic research phases. These versions serve as **reference and archival implementations** for transparency and reproducibility.

The MATLAB legacy component is divided into two categories:

---

### 2.1 Rigorous Optimisation Methods

These solvers are based on deterministic global optimisation and interval-based convex relaxations.

**Algorithms included:**
- **αBB (alpha Branch-and-Bound)**
- **PSO–αBB Hybrid Algorithm**

**Dependency Notice:**  
These implementations rely on **INTLAB**, a proprietary interval arithmetic toolbox developed by Prof. S.M. Rump.  
**INTLAB is *not* included in this repository**, and cannot be redistributed.  
To run the rigorous versions, users must obtain a valid INTLAB licence from the official source.

---

### 2.2 Evolutionary and Population-Based Methods

These solvers represent the evolutionary computing lineage of KANISA.

**Algorithms included:**
- **PSO (Particle Swarm Optimisation)**
- **DE (Differential Evolution)**

These algorithms do not depend on proprietary software and are provided for reference, benchmarking, and historical documentation.

---

## Purpose of the MATLAB Legacy Folder

The MATLAB implementations are included for:

- Preserving the scientific and historical lineage of KANISA  
- Ensuring reproducibility of published research  
- Allowing researchers to replicate earlier experimental environments  
- Providing authoritative baseline implementations for comparison with new Python solvers  

Future versions of KANISA will provide pure Python equivalents of all legacy algorithms.

---
