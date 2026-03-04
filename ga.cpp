#include"utils.h" 
#include <vector>
#include <random>
#include <bitset>
#include <iostream>
#include "ga.h"

using namespace std;
GA::GA(int maxIterations, double tolerance)
    : OptimizationAlgorithm(maxIterations, tolerance) {
        
}

vector<int> GA::encodeToBinary(double value, double min, double max, int bits) {
    value = std::max(min, std::min(value, max));
    double range = max - min;
    double normalized = (value - min) / range;
    int intValue = static_cast<int>(round(normalized * ((1 << bits) - 1)));
    vector<int> binary(bits, 0);
    for (int i = bits - 1; i >= 0; --i) {
        binary[i] = intValue % 2;
        intValue /= 2;
    }
    return binary;
}

vector<double> GA::decode(const vector<int>& binary) {
    vector<double> decodedValues;
    size_t bitIndex = 0;
    const vector<int> bitsPerParam = {17, 15, 18, 7, 10};

    for (size_t i = 0; i < bitsPerParam.size(); ++i) {
        int value = 0;
        for (int j = 0; j < bitsPerParam[i]; ++j) {
            value = (value << 1) | binary[bitIndex++];
        }

        double range = parameterRanges[i].maxValue - parameterRanges[i].minValue;
        double normalized = static_cast<double>(value) / ((1 << bitsPerParam[i]) - 1);
        double decodedValue = parameterRanges[i].minValue + normalized * range;

        decodedValues.push_back(max(parameterRanges[i].minValue,
                                         min(decodedValue, parameterRanges[i].maxValue)));
    }
    return decodedValues;
}

vector<vector<int>> GA::initializePopulation(const vector<double>& initialPoints, int populationSize) {
    vector<vector<int>> population;
    vector<int> seedIndividual;
    for (size_t j = 0; j < parameterRanges.size(); j++) {
        vector<int> binary = encodeToBinary(initialPoints[j], parameterRanges[j].minValue, parameterRanges[j].maxValue, bitsPerParam[j]);
        seedIndividual.insert(seedIndividual.end(), binary.begin(), binary.end());
    }
    population.push_back(seedIndividual);

    for (int i = 1; i < populationSize; i++) {
        vector<int> individual;

        for (size_t j = 0; j < parameterRanges.size(); j++) {
            double perturbation = generateRandom(-0.2, 0.2);
            double value = initialPoints[j] + perturbation * (parameterRanges[j].maxValue - parameterRanges[j].minValue);
            value = max(parameterRanges[j].minValue, min(value, parameterRanges[j].maxValue));
            vector<int> binary = encodeToBinary(value, parameterRanges[j].minValue, parameterRanges[j].maxValue, bitsPerParam[j]);
            individual.insert(individual.end(), binary.begin(), binary.end());
        }
        population.push_back(individual);
    }
    return population;
}


double GA::objectiveFunction(const vector<double>& beta, const vector<array<double, 5>>& data) {
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

pair<vector<double>,double> GA::optimize(
    const vector<array<double, 5>>& data_train,
    const vector<double>& initialPoint) {
    evalCount = 0;
    int populationSize = 50;
    vector<vector<int>> population = initializePopulation(initialPoint, populationSize);

    vector<double> bestBeta;
    double bestFitness = numeric_limits<double>::max();
    double prevBestFitness = numeric_limits<double>::max();

    evaluate(population, data_train);
    update_best(population, data_train, bestBeta, bestFitness);

    int k = 0;
    while (evalCount < static_cast<size_t>(maxIterations)) {
        prevBestFitness = bestFitness;
        
        vector<vector<int>> selected = selection(population, data_train);
        vector<vector<int>> crossedOver = crossover(selected);
        vector<vector<int>> mutated = mutation(crossedOver, 0.1);
        evaluate(mutated, data_train);
        population = createNewPopulation(mutated, data_train, 0.8, 0.1, populationSize);
        update_best(population, data_train, bestBeta, bestFitness);

        k++;
    }
    
    return {bestBeta, static_cast<double>(evalCount)};
}


vector<vector<int>> GA::selection(const vector<vector<int>>& population, const vector<array<double, 5>>& data) {
    vector<double> fitness;
    
    for (const auto& individual : population) {
        vector<double> beta = decode(individual);
        fitness.push_back(objectiveFunction(beta, data));  
    }
    
    vector<double> weights;
    weights.reserve(fitness.size());
    for (double value : fitness) {
        weights.push_back(1.0 / (value + 1e-12));
    }
    double totalWeight = accumulate(weights.begin(), weights.end(), 0.0);

    vector<vector<int>> selectedIndividuals;
    for (int i = 0; i < 2; ++i) {
        double randValue = static_cast<double>(rand()) / RAND_MAX * totalWeight;
        double accumulatedWeight = 0.0;
        
        for (size_t j = 0; j < population.size(); ++j) {
            accumulatedWeight += weights[j];
            if (accumulatedWeight >= randValue) {
                selectedIndividuals.push_back(population[j]);
                break;
            }
        }
    }
    
    return selectedIndividuals;
}
vector<vector<int>> GA::crossover(const vector<vector<int>>& parents) {
    if (parents.size() != 2) {
        throw invalid_argument("Crossover function requires exactly two parents.");
    }

    size_t chromosomeLength = parents[0].size();
    
    int point1 = rand() % chromosomeLength;
    int point2 = rand() % chromosomeLength;
    
    if (point1 > point2) swap(point1, point2);

    vector<int> child1, child2;
    
    child1.insert(child1.end(), parents[0].begin(), parents[0].begin() + point1);
    child2.insert(child2.end(), parents[1].begin(), parents[1].begin() + point1);
    
    child1.insert(child1.end(), parents[1].begin() + point1, parents[1].begin() + point2);
    child2.insert(child2.end(), parents[0].begin() + point1, parents[0].begin() + point2);
    
    child1.insert(child1.end(), parents[0].begin() + point2, parents[0].end());
    child2.insert(child2.end(), parents[1].begin() + point2, parents[1].end());

    return {child1, child2};
}

vector<vector<int>> GA::mutation(vector<vector<int>>& offspring, double mutationRate) {
    size_t chromosomeLength = offspring[0].size();
    
    for (auto& child : offspring) {
        for (size_t i = 0; i < chromosomeLength; ++i) {
            if (rand() / double(RAND_MAX) < mutationRate) {
                child[i] = 1 - child[i]; 
            }
        }

       
        vector<double> decodedChild = decode(child); 
        vector<int> reencodedChild; 
        for (size_t j = 0; j < decodedChild.size(); ++j) {
            const auto& range = parameterRanges[j];
            vector<int> binary = encodeToBinary(decodedChild[j], range.minValue, range.maxValue, bitsPerParam[j]);
            reencodedChild.insert(reencodedChild.end(), binary.begin(), binary.end());
        }
        
        child = reencodedChild; 
    }
    
    return offspring;
}
vector<double> GA::evaluate(const vector<vector<int>>& population, const vector<array<double, 5>>& data) {
    vector<double> fitnessValues;

    for (const auto& individual : population) {
        vector<double> beta = decode(individual);
        
        double fitness = objectiveFunction(beta, data);
        
        fitnessValues.push_back(fitness);
    }

    return fitnessValues;
}
vector<vector<int>> GA::createNewPopulation(
    const vector<vector<int>>& population, 
    const vector<array<double, 5>>& data,
    double crossoverRate,
    double mutationRate,
    int populationSize) {

    vector<vector<int>> newPopulation;
    vector<double> fitness = evaluate(population, data);

    vector<vector<int>> selectedParents = selection(population, data);

    for (int i = 0; i < populationSize / 2; ++i) {
        vector<int> parent1 = selectedParents[i % selectedParents.size()];
        vector<int> parent2 = selectedParents[(i + 1) % selectedParents.size()];

        vector<vector<int>> parents = {parent1, parent2};
        vector<vector<int>> offspring = crossover(parents);

        if (rand() / double(RAND_MAX) < crossoverRate) {
            offspring = crossover(parents);
        } else {
            offspring[0] = parent1;
            offspring[1] = parent2;
        }

        if (rand() / double(RAND_MAX) < mutationRate) {
            mutation(offspring, mutationRate);  
        }

        newPopulation.push_back(offspring[0]);
        newPopulation.push_back(offspring[1]);
    }

    return newPopulation;
}


void GA::update_best(
    const vector<vector<int>>& population,
    const vector<array<double, 5>>& data,
    vector<double>& bestBeta,
    double& bestFitness) {
    
    double currentBestFitness = numeric_limits<double>::max();
    vector<int> bestIndividual;

    for (const auto& individual : population) {
        vector<double> beta = decode(individual);  
        double fitness = objectiveFunction(beta, data);  
        if (fitness < currentBestFitness) {
            currentBestFitness = fitness;
            bestIndividual = individual;  
        }
    }

    if (currentBestFitness < bestFitness) {
        bestFitness = currentBestFitness;
        bestBeta = decode(bestIndividual); 
    }
}
