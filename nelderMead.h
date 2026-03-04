#ifndef NELDERMEAD_H
#define NELDERMEAD_H

#include <vector>
#include <array>
#include "utils.h"
#include "optimization.h"

class NelderMead : public OptimizationAlgorithm {

public:
    NelderMead(int maxIterations, double tolerance);
    std::pair<std::vector<double>,double> optimize(
        const std::vector<std::array<double, 5>>& data_train,
        const std::vector<double>& initialPoint) override;

    std::vector<std::vector<double>> initializeSimplex(const std::vector<double>& initialPoints);

private:
    size_t evalCount = 0;
    double objectiveFunction(const std::vector<double>& beta, const std::vector<std::array<double, 5>>& data);
    void sortSimplex(std::vector<std::vector<double>>& simplex, const std::vector<std::array<double, 5>>& data);
};

#endif
