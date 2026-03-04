#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <array>
#include <string>
#include "optimization.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <numeric>

struct ParameterRange {
    int id;
    std::string name;  
    double minValue;
    double maxValue;
};

struct AlgorithmResults
{
    std::string name;
    std::vector<double> fbest;
    std::vector<double> last_hit;
    AlgorithmResults(const std::string& name, const std::vector<double>& fbest, const std::vector<double>& last_hit)
        : name(name), fbest(fbest), last_hit(last_hit) {}
};

extern std::vector<ParameterRange> parameterRanges;
extern std::array<double, 5> minDataValue;
extern std::array<double, 5> maxDataValue;


class OptimizationAlgorithm;
std::vector<std::array<double, 5>> readData(const std::string& filename);
double predictEnergy(const std::vector<double>& beta, double v, double theta, double T, double P);
double generateRandom(double lower, double upper);
void clampParameters(std::vector<double>& beta);
void logResults(const std::string& filename, const std::vector<double>& beta, double mse, double k, double experiment);
void initializeLogFile(const std::string& filename);
void generateAndSaveInitialPoints(const std::vector<ParameterRange>& parameterRanges, const std::string& filename);
std::vector<std::vector<double>> loadInitialPoints(const std::string& filename);
std::vector<double> runExperiment(OptimizationAlgorithm& optimizer, const std::vector<std::array<double, 5>>& data_train,  const std::vector<std::array<double, 5>>& data_test,
                   const std::vector<std::vector<double>>& initialPoints, const std::string& resultFile);
void calculateMinMax(const std::vector<std::array<double, 5>>& data);
void normalizeData(std::vector<std::array<double, 5>>& data);
void denormalizeData(std::vector<std::array<double, 5>>& data);
void normalizeBeta(std::vector<double>& beta);
void denormalizeBeta(std::vector<double>& beta) ;
void computeStatistics(const std::vector<double>& data, double& mean, double& median, double& stdev, double& min, double& max);
void readResults(const std::string& filename, std::vector<double>& data, int columnIndex);
void exportToCSV(const std::vector<AlgorithmResults>& algorithms, const std::string& outputFile);
void exportStatisticsToText(const std::vector<AlgorithmResults>& algorithms, const std::string& outputFile);
void generateSignificanceTable(const std::vector<std::string>& algorithms, const std::vector<std::vector<double>>& fbestSamples, const std::string& outputFile);
double computeWilcoxonPValue(const std::vector<double>& sample1, const std::vector<double>& sample2);
void writeTestResultsToFile(const std::string& outputFile, const std::vector<std::vector<double>>& mseResults);
#endif
