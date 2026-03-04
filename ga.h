#ifndef GA_H
#define GA_H

#include <vector>
#include <array>
#include "utils.h"
#include "optimization.h"

using namespace std;

class GA : public OptimizationAlgorithm {
public:
    const vector<int> bitsPerParam = {17, 15, 18, 7, 10};
    GA(int maxIterations, double tolerance);
    bool usesNormalized() const override { return false; }
    pair<vector<double>,double> optimize(
        const vector<array<double, 5>>& data_train,
        const vector<double>& initialPoint) override;
private:
    size_t evalCount = 0;
    double objectiveFunction(const vector<double>& beta, const vector<array<double, 5>>& data);
    vector<int> encodeToBinary(double value, double min, double max, int bits);
    vector<vector<int>> initializePopulation(const vector<double>& initialPoints, int populationSize);
    vector<vector<int>> crossover(const vector<vector<int>>& population);
    vector<vector<int>> mutation(vector<vector<int>>& offspring, double mutationRate);
    vector<vector<int>> selection(const vector<vector<int>>& population, const vector<array<double, 5>>& data);
    vector<vector<int>> createNewPopulation(
        const vector<vector<int>>& population,
        const vector<array<double, 5>>& data,
        double crossoverRate,
        double mutationRate,
        int populationSize);
    vector<double> evaluate(const vector<vector<int>>& population, const vector<array<double, 5>>& data);
    void update_best(const vector<vector<int>>& population,
    const vector<array<double, 5>>& data,
    vector<double>& bestBeta,
    double& bestFitness);
    vector<double> decode(const vector<int>& binary);
};

#endif
