#include <vector>
#include <array>
#include <iostream>
#include <random>
#include <limits>
#include "pso.h"
#include "utils.h"

PSO::PSO(int maxIterations, double tolerance)
    : OptimizationAlgorithm(maxIterations, tolerance) {
       
    }

double PSO::objectiveFunction(const std::vector<double>& beta, const std::vector<std::array<double, 5>>& data) {
    evalCount++;
    double mse = 0.0;
    std::vector<double> denormalizedBeta = beta;
    denormalizeBeta(denormalizedBeta);
    std::vector<std::array<double, 5>> normalizedData = data;
    for (const auto& row : normalizedData) {
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

std::pair<std::vector<double>,double> PSO::optimize(const std::vector<std::array<double, 5>>& data_train, const std::vector<double>& initialPoint) {
    evalCount = 0;
    const int swarmSize = 30;
    const double c1 = 2.05, c2 = 2.05, inertiaWeight = 0.729, maxVelocity = 0.3;

    
    std::vector<std::vector<double>> swarm(swarmSize);
    std::vector<std::vector<double>> velocities(swarmSize);
    std::vector<std::vector<double>> bestPositions(swarmSize);
    std::vector<double> bestFitnesses(swarmSize, std::numeric_limits<double>::max());
    globalBestPosition = initialPoint;
    double globalBestFitness = std::numeric_limits<double>::max();
    double prevBestFitness = globalBestFitness;
    int iter = 0;
    swarm[0] = initialPoint;
    velocities[0] = initializeVelocity(initialPoint);
    bestPositions[0] = swarm[0];
    bestFitnesses[0] = objectiveFunction(swarm[0], data_train);
    globalBestFitness = bestFitnesses[0];
    globalBestPosition = swarm[0];

    for (int i = 1; i < swarmSize; ++i) {
        swarm[i] = initialSpace(initialPoint);
        velocities[i] = initializeVelocity(initialPoint);
        bestPositions[i] = swarm[i];
        bestFitnesses[i] = objectiveFunction(swarm[i], data_train);
        if (bestFitnesses[i] < globalBestFitness) {
            globalBestFitness = bestFitnesses[i];
            globalBestPosition = swarm[i];
        }
    }

    while (evalCount < static_cast<size_t>(maxIterations)) {
        prevBestFitness = globalBestFitness;
        for (int i = 0; i < swarmSize; ++i) {
           
            velocities[i] = updateVelocity(velocities[i], swarm[i], bestPositions[i], globalBestPosition, c1, c2, inertiaWeight);
            checkBounds(velocities[i], maxVelocity);

            
            swarm[i] = updatePosition(swarm[i], velocities[i]);

            
            double fitness = objectiveFunction(swarm[i], data_train);

           
            if (fitness < bestFitnesses[i]) {
                bestFitnesses[i] = fitness;
                bestPositions[i] = swarm[i];
            }

            
            if (fitness < globalBestFitness) {
                globalBestFitness = fitness;
                globalBestPosition = swarm[i];
        }
        iter++;
    }

    }

    return {globalBestPosition, static_cast<double>(evalCount)};
}

std::vector<double> PSO::initializeVelocity(const std::vector<double>& initialSpace) {
    std::vector<double> velocity(initialSpace.size());
    for (size_t i = 0; i < initialSpace.size(); ++i) {
        velocity[i] = generateRandom(-0.01, 0.01);
    }
    return velocity;
}

std::vector<double> PSO::initialSpace(const std::vector<double>& initialPoint) {
    std::vector<double> candidate = initialPoint;
    for (size_t i = 0; i < candidate.size(); ++i) {
        candidate[i] += generateRandom(-0.1, 0.1);
    }
    clampParametersNormalized(candidate);
    return candidate;
}

std::vector<double>PSO::updateVelocity(const std::vector<double>& velocity, const std::vector<double>& position, const std::vector<double>& bestPosition, const std::vector<double>& globalBestPosition, double c1, double c2, double inertiaWeight) {
    
    std::vector<double> newVelocity(velocity.size());
    for (size_t i = 0; i < velocity.size(); ++i) {
        double r1 = generateRandom(0.0, 1.0);
        double r2 = generateRandom(0.0, 1.0);
        newVelocity[i] = inertiaWeight * velocity[i]+ c1 * r1 * (bestPosition[i] - position[i]) + c2 * r2 * (globalBestPosition[i] - position[i]);
    }
    return newVelocity;
}

std::vector<double> PSO::updatePosition(const std::vector<double>& position, const std::vector<double>& velocity) {
    std::vector<double> newPosition(position.size());
    for (size_t i = 0; i < position.size(); ++i) {
        newPosition[i] = position[i] + velocity[i];
    }
    clampParametersNormalized(newPosition);
    return newPosition;
}

void PSO::checkBounds(std::vector<double>& velocity, double maxVelocity) {
    for (auto& v : velocity) {
        if (v > maxVelocity) v = maxVelocity;
        else if (v < -maxVelocity) v = -maxVelocity;
    }
}
