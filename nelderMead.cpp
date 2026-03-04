#include "nelderMead.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>

using namespace std;

NelderMead::NelderMead(int maxIterations, double tolerance)
    : OptimizationAlgorithm(maxIterations, tolerance) {
}
double NelderMead::objectiveFunction(const vector<double> &beta, const vector<array<double, 5>> &data)
{
    evalCount++;
    double mse = 0.0;
    vector<double> denormalizedBeta = beta;
    vector<array<double, 5>> normilizedData = data;
    for (const auto& row : normilizedData) {
        double v = row[0];
        double theta = row[1];
        double T = row[2];
        double P = row[3];
        double E = row[4]; 
        double predicted = predictEnergy(denormalizedBeta, v, theta, T, P);
        mse += (-E + predicted) * (-E + predicted);
    }
    
    return mse / data.size();
}
vector<vector<double>> NelderMead::initializeSimplex(const vector<double>& initialPoints) {
    int numParams = initialPoints.size();  
    vector<vector<double>> simplex(numParams + 1, vector<double>(numParams));

    for (int j = 0; j < numParams; ++j) {
        simplex[0][j] = initialPoints[j];
    }

    for (int i = 1; i < numParams + 1; ++i) {
        for (int j = 0; j < numParams; ++j) {
            double span = parameterRanges[j].maxValue - parameterRanges[j].minValue;
            simplex[i][j] = initialPoints[j] + generateRandom(-0.1, 0.1) * span;
        }
        clampParameters(simplex[i]);
    }

    return simplex;
}

void NelderMead::sortSimplex(vector<vector<double>>& simplex, const vector<array<double, 5>>& data) {
    sort(simplex.begin(), simplex.end(), [&](const vector<double>& a, const vector<double>& b) {
        return objectiveFunction(a, data) < objectiveFunction(b, data);
    });
}

pair<vector<double>,double> NelderMead::optimize(const vector<array<double, 5>>& data, const vector<double>& initialPoint) {
    evalCount = 0;
    double reflection_coefficient = 1.0;
    double expansion_coefficient = 2.0;
    double contraction_coefficient = 0.5;

    auto simplex = initializeSimplex(initialPoint);
    int iter = 0;

    for (iter = 0; iter < maxIterations; iter++) {
        if (evalCount >= static_cast<size_t>(maxIterations)) {
            break;
        }

        sortSimplex(simplex, data);

        double maxDiff = 0.0;
        for (size_t i = 1; i < simplex.size(); ++i) { 
            double diff = abs(objectiveFunction(simplex[i], data) - objectiveFunction(simplex[0], data));
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        if (maxDiff < this->tolerance) {
            break;
        }

        vector<double> centroid(simplex[0].size(), 0.0);  
        for (int i = 0; i < static_cast<int>(simplex.size()) - 1; i++) {
            for (int j = 0; j < simplex[i].size(); j++) {
                centroid[j] += simplex[i][j];
            }

        }

        for (int j = 0; j < centroid.size(); j++) {
            centroid[j] /= (simplex.size() - 1);
        }

        vector<double> reflection(simplex[0].size());
        for (int j = 0; j < simplex[0].size(); j++) {
            reflection[j] = (1 + reflection_coefficient) * centroid[j] - reflection_coefficient * simplex.back()[j];
        }
        clampParameters(reflection);
        double reflectedValue = objectiveFunction(reflection, data);
        double worstValue = objectiveFunction(simplex.back(), data);

        
        double bestValue = objectiveFunction(simplex[0], data);
        double secondWorstValue = objectiveFunction(simplex[simplex.size() - 2], data);

        if (reflectedValue < bestValue) {
          
            vector<double> expansion(simplex[0].size()); 
            for (int j = 0; j < simplex[0].size(); j++) {
                expansion[j] = (1 + expansion_coefficient) * centroid[j] - expansion_coefficient * simplex.back()[j];
            }
            clampParameters(expansion);
            double expandedValue = objectiveFunction(expansion, data);
            

            if (expandedValue < reflectedValue) {
                simplex.back() = expansion;
                
            } else {
                simplex.back() = reflection;
                
            }
        } else if (reflectedValue < secondWorstValue) {
            simplex.back() = reflection;
        } else {
            if (reflectedValue < worstValue) {
                simplex.back() = reflection;
            }

                vector<double> contraction(simplex[0].size()); 
                for (int j = 0; j < simplex[0].size(); j++) {
                    contraction[j] = (1 + contraction_coefficient) * centroid[j] - contraction_coefficient * simplex.back()[j];
                }
                clampParameters(contraction);
                double contractedValue = objectiveFunction(contraction, data);
                

                if (contractedValue < worstValue) {
                    simplex.back() = contraction;
                   
                } else {
                   
                    for (int i = 1; i < simplex.size(); i++) {
                        for (int j = 0; j < simplex[i].size(); j++) {
                            simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                        }
                        clampParameters(simplex[i]);
                    }
                  
                }
        }

    }

    return {simplex[0], static_cast<double>(evalCount)};
}

