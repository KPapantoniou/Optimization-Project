#ifndef NEWTONTR_H
#define NEWTONTR_H

#include <vector>
#include <array>
#include "utils.h"
#include "optimization.h"

using namespace std;

class NewtonTr : public OptimizationAlgorithm  {
public:
    NewtonTr(int maxIterations, double tolerance);
    pair<vector<double>,double> optimize(
        const vector<array<double, 5>>& data_train,
        const vector<double>& initialPoint) override;
    static vector<double> gradient(const vector<double>& x, const vector<array<double, 5>>& data);
    static vector<vector<double>> hessian(const vector<double>& beta, const vector<array<double, 5>>& data);
    static bool isPositiveDefinite(const vector<vector<double>>& matrix);
    static void invertMatrix(const vector<vector<double>>& matrix, vector<vector<double>>& inverse);
    static vector<double> addVectors(const vector<double>& v1, const vector<double>& v2);
    static void hessianAproximation( vector<vector<double>>& matrix);
    static vector<double>matrixVectorMultiplication(const vector<vector<double>>& matrix, const vector<double>& vector);
    static double norm(const vector<double>& vector);
    static double dotProduct(const vector<double>& a, const vector<double>& b);
    static vector<double> negateVector(const vector<double>& vec);
    static bool quadraticForm(const vector<double>& v, const vector<vector<double>>& H, const vector<double>& u);
    static void luDecomposition(const vector<vector<double>>&matrix, vector<vector<double>>& L,vector<vector<double>>& U);
    static void forwardSubstitution(const vector<vector<double>>& L, const vector<vector<double>> I, vector<vector<double>>& Y);
    static void backwardSubstitution(const vector<vector<double>>& U, const vector<vector<double>> Y, vector<vector<double>>& X) ;

private:
    size_t evalCount = 0;
    vector<double>doglegDirection(const vector<double> gl, const vector<vector<double>> invertBMatrix, double delta_k);
    double modelFunction(double fk, const vector<double>& g_k, const vector<vector<double>>& B_k, const vector<double>& p_k);
    double solveTau(const vector<double> pU, const vector<double> pB, double delta_k);
    
    double objectiveFunction(const vector<double>& beta, const vector<array<double, 5>>& data);
    
    
};

#endif
