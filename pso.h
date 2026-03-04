#ifndef PSO_H
#define PSO_H

#include <vector>
#include <array>
#include "utils.h"
#include "optimization.h"



class PSO : public OptimizationAlgorithm {

public:
    std::vector<double> globalBestPosition;
    PSO(int maxIterations, double tolerance);
    std::pair<std::vector<double>,double> optimize(
        const std::vector<std::array<double, 5>>& data_train,
        const std::vector<double>& initialPoint) override;
private:
    size_t evalCount = 0;
    double objectiveFunction(const std::vector<double>& beta, const std::vector<std::array<double, 5>>& data);
    std::vector<double> initialSpace(const std::vector<double>& initialPoint);
    std::vector<double> initializeVelocity(const std::vector<double>& intiliaSpace);
    std::vector<double> updateVelocity(const std::vector<double>& velocity, const std::vector<double>& position, const std::vector<double>& bestPosition, const std::vector<double>& globalBestPosition, double c1, double c2, double inertiaWeight);
    std::vector<double> updatePosition(const std::vector<double>& position, const std::vector<double>& velocity);
    void checkBounds(std::vector<double>& velocity, double maxVelocity);
};  

#endif
