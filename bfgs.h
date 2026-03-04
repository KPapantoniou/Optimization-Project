#ifndef BFGS_H
#define BFGS_H

#include <vector>
#include <array>
#include "utils.h"
#include "optimization.h"
#include "newtonTr.h"

class BFGS : public OptimizationAlgorithm {
public:
    BFGS(int maxIterations, double tolerance);
    std::pair<std::vector<double>,double> optimize(
        const std::vector<std::array<double, 5>>& data_train,
        const std::vector<double>& initialPoint) override;
    static std::vector<double>scalarMultiplyVector(double scalar, const std::vector<double>& v);
    static void printVector(const std::vector<double>& vec);


    
private:
    size_t evalCount = 0;
    double objectiveFunction(const std::vector<double>& beta, const std::vector<std::array<double, 5>>& data);
    double lineSearch(const std::vector<double>& x, const std::vector<double>& p, const std::vector<std::array<double, 5>>& data);
    std::vector<std::vector<double>>BFGSUpdate(std::vector<std::vector<double>> B, std::vector<double> s, std::vector<double> y);
    std::vector<std::vector<double>> outerProduct(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<std::vector<double>> addMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
    std::vector<std::vector<double>> subtractMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
    std::vector<std::vector<double>>matrixMultiplication(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
    std::vector<std::vector<double>>scalarMultiplyMatrix(const std::vector<std::vector<double>>& matrix, double scalar);
    std::vector<std::vector<double>> negateMatrix(const std::vector<std::vector<double>>& matrix);
    double interpolate(double alphaL, double alphaH);
    double zoom(double alphaL, double alphaH, double phi_zero, double phi_prime_zero, double c1, double c2, const std::vector<std::array<double, 5>>& data, const std::vector<double>& beta, const std::vector<double>& direction);
};  

#endif
