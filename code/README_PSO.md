# Particle Swarm Optimization Homework - Spring 2026

## Overview
This notebook implements a comprehensive Particle Swarm Optimization (PSO) study comparing 10 different PSO variants on two challenging optimization benchmarks.

## File Location
- **Notebook**: `/sessions/zen-busy-maxwell/mnt/HW5/code/PSO_Homework.ipynb`
- **Results Directory**: `/sessions/zen-busy-maxwell/mnt/HW5/code/PSO_Results/`

## Contents

### 1. Objective Functions
The notebook implements two benchmark functions:

#### Eggholder Function (2D)
- Mathematical form: `f(x₁, x₂) = -(x₂ + 47) * sin(√|x₂ + x₁/2 + 47|) - x₁ * sin(√|x₁ - (x₂ + 47)|)`
- Domain: [-512, 512]²
- Global minimum: f(512, 404.2319) = -959.6407
- Characteristic: Highly multimodal with many local optima

#### Schwefel Function (4D)
- Mathematical form: `f(x) = 418.98n + Σᵢ[-xᵢ * sin(√|xᵢ|)]`
- Domain: [-512, 512]⁴
- Global minimum: f(420.97, ..., 420.97) = 0
- Characteristic: Highly deceptive with misleading local minima

### 2. PSO Implementation
Complete PSO class with support for:
- **Topologies**: Star (global best) and Ring (neighborhood best)
- **Models**: Full, Cognition-only, Social-only, Selfless
- **Velocity Control**: Inertia weight and Constriction factor
- **Boundary Handling**: Clamping with velocity reset

Key parameters:
- Population size: 30 (Eggholder default), 50 (Schwefel default)
- Max iterations: 500
- Velocity limits: v_max = 512
- Inertia: Linear decrease from 0.9 to 0.4
- Constriction: K = 2/|2 - φ - √(φ² - 4φ)| for φ = φ₁ + φ₂

### 3. Experimental Design

#### 10 PSO Variants
1. **Variant 1**: Full model, Star topology, Base (φ₁=φ₂=2)
2. **Variant 2**: Cognition only (φ₂=0)
3. **Variant 3**: Social only (φ₁=0)
4. **Variant 4**: Selfless (g≠i, excludes self)
5. **Variant 5**: Ring topology, Full model
6. **Variant 6**: Inertia enabled
7. **Variant 7**: Constriction enabled (φ₁=φ₂=2.05)
8. **Variant 8**: Inertia + Constriction (φ₁=φ₂=2.05)
9. **Variant 9a**: More particles (60 for Eggholder, 100 for Schwefel)
10. **Variant 9b**: Fewer particles (15 for Eggholder, 25 for Schwefel)

#### Seeds
All experiments use seeds: [1, 2, 3, 4, 5]
- **Total experiments per function**: 10 variants × 5 seeds = 50
- **Total experiments overall**: 100 (both functions)

### 4. Notebook Structure

#### Section 1: Imports and Setup
- NumPy, Matplotlib, Pandas, Seaborn configuration
- Environment validation

#### Section 2: Objective Functions
- Eggholder function implementation
- Schwefel function implementation
- Function testing and validation

#### Section 3: PSO Class
- Complete PSO algorithm implementation
- Support for all topology/model combinations
- Velocity updates with inertia and constriction
- Boundary handling

#### Section 4: Experiment Configuration
- Variant definitions
- Seed specification
- Parameter mapping

#### Section 5-6: Eggholder Experiments
- Run 50 experiments (10 variants × 5 seeds)
- Results summary table
- Convergence plots
- Performance analysis

#### Section 7-8: Schwefel Experiments
- Run 50 experiments with SAME seeds (critical for bonus)
- Results summary table
- Convergence plots
- Performance analysis

#### Section 9: Convergence Analysis
- Individual convergence curves for each variant
- Visualization of optimization progress

#### Section 10: Final Summary
- Cross-function performance comparison
- Best-performing variants identification
- Statistical summary

## Execution Guide

### Running the Notebook

1. **Open in Jupyter**:
   ```bash
   jupyter notebook /sessions/zen-busy-maxwell/mnt/HW5/code/PSO_Homework.ipynb
   ```

2. **Run All Cells**: Use Jupyter's "Run All Cells" or Kernel → Restart & Run All

3. **Expected Runtime**: ~5-15 minutes for all 100 experiments (depends on hardware)

### Output

The notebook generates:
- Comprehensive results tables for each function
- Convergence plots showing optimization progress
- Boxplots comparing variant performance
- Performance metrics (Mean, Std, Best, Worst)

## Key Results

### Eggholder Function (2D)
- Global optimum: -959.6407
- Default particles: 30
- Best variant: Typically Inertia+Constriction

### Schwefel Function (4D)
- Global optimum: 0.0
- Default particles: 50
- Best variant: Typically Inertia+Constriction

## Technical Details

### PSO Algorithm Flow
1. Initialize particles with random positions and velocities
2. Evaluate fitness for all particles
3. Update personal best and global best
4. For each iteration:
   - Update velocity using cognitive + social components
   - Apply inertia/constriction if enabled
   - Clamp velocity to limits
   - Update position
   - Handle boundary violations
   - Evaluate new fitness
   - Update personal/global bests

### Boundary Handling
- Positions clamped to [-512, 512]
- Velocity reset to 0 on violating dimension
- Ensures algorithm respects search space

### Velocity Update Equations
- **Base**: v = v + c₁r₁(p_best - x) + c₂r₂(g_best - x)
- **With Inertia**: v = w*v + c₁r₁(p_best - x) + c₂r₂(g_best - x)
- **With Constriction**: v = K[v + c₁r₁(p_best - x) + c₂r₂(g_best - x)]
- **Both**: v = K[w*v + c₁r₁(p_best - x) + c₂r₂(g_best - x)]

## Notes

- Seeds are identical for both functions to enable fair comparison
- Ring topology uses left-right neighborhood (3 neighbors)
- Selfless model excludes current particle from neighborhood best
- All results stored in local memory for analysis
- Convergence history tracked for visualization

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. ICNN'95 - International Conference on Neural Networks.
- Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space.
- Trelea, I. C. (2003). The particle swarm optimization algorithm: convergence analysis and parameter selection.

