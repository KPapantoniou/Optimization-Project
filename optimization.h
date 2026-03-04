#ifndef OPTIMIZATION_ALGORITHM_H
#define OPTIMIZATION_ALGORITHM_H

#include <vector>
#include <array>
#include "utils.h" 
class OptimizationAlgorithm {
public:

    OptimizationAlgorithm(int maxIterations, double tolerance);

    virtual std::pair<std::vector<double>,double> optimize(
        const std::vector<std::array<double, 5>>& data_train,
        const std::vector<double>& initialPoint) = 0;

    virtual ~OptimizationAlgorithm() {}

protected:
    int maxIterations;    
    double tolerance;     
};

#endif 
