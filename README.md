# Optimization Bonus Project 2024

This project fits a simple energy prediction model by optimizing five parameters using multiple algorithms and comparing their performance. It runs 30 randomized experiments per algorithm, logs train/test mean squared error (MSE), and generates summary statistics plus a significance table.

## Model
Given inputs `v`, `theta`, `T`, `P`, the energy is predicted as:

```
E = b1 * v^2 + b2 * sin(theta) + b3 * exp(b4 * T) + b5 * log(P)
```

Notes:
- `theta` is converted from degrees to radians when loading the data.
- Parameter ranges are fixed in code (see `utils.cpp`).

## Algorithms
- Nelder-Mead
- Newton Trust Region
- BFGS (Wolfe line search)
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)

## Data
- `data_train.txt` and `data_test.txt` contain rows of `v theta T P E`.
- The program reads the files, converts `theta` to radians, and uses the last column as the target energy.

## How to Build and Run

Requirements:
- CMake >= 3.16
- C++17 compiler
- Python 3 (only for the boxplot)

Build and run:

```
cmake -S . -B build
cmake --build build
cmake --build build --target run
```

Or run directly after building:

```
./build/optimization_project
```

To remove generated output files:

```
cmake --build build --target clean_generated
```

## Output Files
When you run the project, the following artifacts are created in the build directory:

- `initial_points.txt`: 30 randomized starting points.
- `output_*_train.txt`: per-run training MSE and parameters for each algorithm.
- `output_test.txt`: per-run test MSE for each algorithm.
- `results.csv`: per-run test MSE values (one column per algorithm).
- `statistics.txt`: summary statistics for `fbest` (test MSE) and `last-hit`.
- `significance.txt`: pairwise Wilcoxon rank-sum p-values with directional hints.
- `solution_boxplot.png`: boxplot of test MSE (requires Python 3).

## Results (latest run in `build/`)

Summary statistics from `build/statistics.txt`:

```
fbest (test MSE)
- Mean   : NelderMead=116.142  NewtonTR=127.392  BFGSWolfe=408716  GA=115.635  PSO=114.551
- Median : NelderMead=110.175  NewtonTR=123.166  BFGSWolfe=33795.7 GA=115.184  PSO=110.052
- St.dev.: NelderMead=10.6278  NewtonTR=21.3289  BFGSWolfe=698810  GA=3.40928  PSO=7.83068
- Min    : NelderMead=110.052  NewtonTR=110.172  BFGSWolfe=125.565 GA=110.794  PSO=110.052
- Max    : NelderMead=148.106  NewtonTR=233.109  BFGSWolfe=2.45236e+06 GA=125.243 PSO=133.461

last-hit (iteration of best result)
- Mean   : NelderMead=15750.7  NewtonTR=99568.4  BFGSWolfe=100004  GA=100058  PSO=100020
- Median : NelderMead=15321    NewtonTR=100000   BFGSWolfe=100001  GA=100058  PSO=100020
- St.dev.: NelderMead=5613.63  NewtonTR=2324.24  BFGSWolfe=8.21848 GA=0       PSO=0
- Min    : NelderMead=6038     NewtonTR=87052    BFGSWolfe=100000  GA=100058  PSO=100020
- Max    : NelderMead=33906    NewtonTR=100000   BFGSWolfe=100031  GA=100058  PSO=100020
```

Pairwise significance (Wilcoxon rank-sum) from `build/significance.txt`:

```
NelderMead vs NewtonTR  p=0.000116499 (+)
NelderMead vs BFGSWolfe p=5.9271e-11 (+)
NelderMead vs GA        p=0.00194192 (-)
NelderMead vs PSO       p=0.00799816 (-)

NewtonTR vs BFGSWolfe   p=1.46431e-10 (+)
NewtonTR vs GA          p=3.36814e-05 (-)
NewtonTR vs PSO         p=6.92888e-06 (-)

BFGSWolfe vs GA         p=3.01986e-11 (-)
BFGSWolfe vs PSO        p=1.85064e-11 (-)

GA vs PSO               p=0.00112729 (-)
```

Legend:
- `+` means the row algorithm has lower mean MSE than the column algorithm.
- `-` means the row algorithm has higher mean MSE than the column algorithm.
- The values above are from the latest run in `build/` and will change if you rerun.

## Notes
- The program runs the five algorithms in parallel threads.
- BFGS can be unstable on this objective for some random initializations, which explains the large MSE outliers in the current run.

## Project Files
- `CMakeLists.txt`: build configuration.
- `main.cpp`: experiment runner and result aggregation.
- `utils.*`: data loading, normalization, metrics, and reporting.
- `nelderMead.*`, `newtonTr.*`, `bfgs.*`, `ga.*`, `pso.*`: algorithm implementations.
