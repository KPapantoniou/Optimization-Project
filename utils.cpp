#include "utils.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include "optimization.h"

#define RUNS 30

using namespace std;

    vector<ParameterRange> parameterRanges = {
    {0, "β1", 10, 100},     
    {1, "β2", -10, 10},     
    {2, "β3", 50, 200},     
    {3, "β4", 0.01, 0.1},  
    {4, "β5", 0.1, 1}      
};

array<double, 5> minDataValue;
array<double, 5> maxDataValue;
vector<vector<double>> mseResults(30, vector<double>(5, 0.0));  


vector<array<double, 5>> readData(const string& filename) {
    ifstream file(filename);
    vector<array<double, 5>> data;
    double v, theta, T, P, E;
    while (file >> v >> theta >> T >> P >> E) {
        double theta_rad = theta * M_PI / 180.0;
        data.push_back({v, theta_rad, T, P, E});
    }
    return data;
}

double predictEnergy(const vector<double>& beta, double v, double theta, double T, double P) {
    return beta[0] * v * v + beta[1] * sin(theta) + beta[2] * exp(beta[3] * T) + beta[4] * log(P);
}

double generateRandom(double lower, double upper) {
    static thread_local random_device rd;
    static thread_local mt19937 gen(rd());
    uniform_real_distribution<> dis(lower, upper);
    return dis(gen);
}

void clampParameters(vector<double>& beta) {
    for (size_t i = 0; i < beta.size() && i < parameterRanges.size(); ++i) {
        beta[i] = max(parameterRanges[i].minValue, min(beta[i], parameterRanges[i].maxValue));
    }
}

void logResults(const string& filename, const vector<double>& parameters, double mse, double k, double experiment) {
    ofstream logFile(filename, ios::app); 
    if (logFile.is_open()) {
        logFile<< experiment+1 << " ";
        logFile << mse << " ";
        logFile << k << " ";
        for(const auto& parameter : parameters) {
            logFile << parameter << " ";
        }
        logFile<<"\n";
        logFile << endl;
        logFile.close();
    } else {
        cerr << "Failed to open log file!" << endl;
    }
}
void initializeLogFile(const string& filename) {
    ofstream logFile(filename, ios::out | ios::trunc); 
    if (logFile.is_open()) {
        logFile.close(); 
    } else {
        cerr << "Failed to initialize log file: " << filename << endl;
    }
}
void generateAndSaveInitialPoints(const vector<ParameterRange>& parameterRanges, const string& filename) {
    ofstream outfile(filename);
    vector<double> initialPoint(parameterRanges.size());
    for (int i = 0; i < RUNS; ++i) {
        for (size_t j = 0; j < parameterRanges.size(); ++j) {
            initialPoint[j] = generateRandom(parameterRanges[j].minValue, parameterRanges[j].maxValue);
        }

        for (const auto& value : initialPoint) {
            outfile << value << " ";
        }
        outfile << "\n";
    }
    outfile.close();
}
vector<vector<double>> loadInitialPoints(const string& filename) {
    ifstream infile(filename);
    vector<vector<double>> initialPoints(RUNS, vector<double>(5));
    
    for (int i = 0; i < RUNS; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            infile >> initialPoints[i][j];
        }
    }

    infile.close();
    return initialPoints;
}

vector<double> runExperiment(OptimizationAlgorithm& optimizer, const vector<array<double, 5>>& data_train, const vector<array<double, 5>>& data_test,
                    const vector<vector<double>>& initialPoints, const string& resultFile) {

    vector<double> mse_test_results;
    mse_test_results.reserve(RUNS);
    for (int experiment = 0; experiment <RUNS; ++experiment) {
        vector<double> initialPoint = initialPoints[experiment];
        clampParameters(initialPoint);

        auto result = optimizer.optimize(data_train, initialPoint);
        auto bestParameters = result.first;
        auto k = result.second;
        clampParameters(bestParameters);
        double mse_train = 0.0;

        for (const auto& row : data_train) {
            double predicted = predictEnergy(bestParameters, row[0], row[1], row[2], row[3]);
            mse_train += (-row[4] + predicted) * (-row[4] + predicted);
        }
        mse_train /= data_train.size();

        logResults(resultFile, bestParameters, mse_train, k, experiment);
        double mse_test = 0.0;
        for (const auto& row : data_test) {
            double predicted = predictEnergy(bestParameters, row[0], row[1], row[2], row[3]);
            mse_test += (predicted - row[4]) * (predicted - row[4]);
        }
        mse_test /= data_test.size();
        mse_test_results.push_back(mse_test);
    }
    return mse_test_results;

}



void calculateMinMax(const vector<array<double, 5>>& data) {
    for (size_t i = 0; i < 5; ++i) {
        minDataValue[i] = numeric_limits<double>::max();
        maxDataValue[i] = numeric_limits<double>::lowest();
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            minDataValue[i] = min(minDataValue[i], row[i]);
            maxDataValue[i] = max(maxDataValue[i], row[i]);
        }
    }
}

void normalizeData(vector<array<double, 5>>& data) {
    calculateMinMax(data);

    for (auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i == 3) {
                row[i] = (row[i] - minDataValue[i]) / (maxDataValue[i] - minDataValue[i]) + 1e-8;
            } else {
                row[i] = (row[i] - minDataValue[i]) / (maxDataValue[i] - minDataValue[i]);
            }
        }
    }
}


void denormalizeData(vector<array<double, 5>>& data) {
    for (auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i == 3) {
                row[i] = row[i] * (maxDataValue[i] - minDataValue[i]) + minDataValue[i] - 1e-8;
            } else {
                row[i] = row[i] * (maxDataValue[i] - minDataValue[i]) + minDataValue[i];
            }
        }
    }
}

void calculateMinMaxBeta(const vector<double>& beta, double& minBeta, double& maxBeta) {
    minBeta = *min_element(beta.begin(), beta.end());
    maxBeta = *max_element(beta.begin(), beta.end());
}

void normalizeBeta(vector<double>& beta) {
    vector<double> minBeta = { 10.0, -10.0, 50.0, 0.01, 0.10 }; 
    vector<double> maxBeta = { 100.0, 10.0, 200.0, 0.1, 1.0 };

    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] = (beta[i] - minBeta[i]) / (maxBeta[i] - minBeta[i]);
    }
}

void denormalizeBeta(vector<double>& beta) {
    vector<double> minBeta = { 10.0, -10.0, 50.0, 0.01, 0.10 };  
    vector<double> maxBeta = { 100.0, 10.0, 200.0, 0.1, 1.0 };

    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] = beta[i] * (maxBeta[i] - minBeta[i]) + minBeta[i];
    }
}

void computeStatistics(const vector<double>& data, double& mean, double& median, double& stdev, double& min, double& max) {
    int n = data.size();
    mean = accumulate(data.begin(), data.end(), 0.0) / n;

    vector<double> sorted_data = data;
    sort(sorted_data.begin(), sorted_data.end());
    median = (n % 2 == 0) ? (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0 : sorted_data[n / 2];

    double sum_of_squares = accumulate(data.begin(), data.end(), 0.0,
                                            [mean](double acc, double val) { return acc + (val - mean) * (val - mean); });
    stdev = sqrt(sum_of_squares / n);

    min = *min_element(data.begin(), data.end());
    max = *max_element(data.begin(), data.end());
}
void readResults(const string& filename, vector<double>& results, int columnIndex) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        istringstream ss(line);
        double value;
        int currentIndex = 0;

        while (ss >> value) {
            if (currentIndex == columnIndex) {  
                results.push_back(value);
            }
        
            currentIndex++;
        }
    }

    file.close();
}


void writeTestResultsToFile(const string& outputFile, const vector<vector<double>>& mseResults) {
    ofstream file(outputFile);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << outputFile << endl;
        return;
    }

    for (size_t i = 0; i < RUNS; ++i) {
        file << i + 1;
        for (size_t j = 0; j < 5; ++j) {
            file << " " << mseResults[i][j];
        }
        file << "\n";
    }

    file.close();
}

void exportToCSV(const vector<AlgorithmResults>& algorithms, const string& outputFile) {
    ofstream file(outputFile);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << outputFile << endl;
        return;
    }
    for (size_t i = 0; i < algorithms.size(); ++i) {
            file << algorithms[i].name;
            if (i < algorithms.size()) {
                file << ",";
            }
        }
    file << endl;

    size_t max_rows = 0;
    for (const auto& algo : algorithms) {
        max_rows = max(max_rows, algo.fbest.size());
    }
    for(size_t i = 0; i<max_rows; ++i){
        for(const auto& algo : algorithms){
            if(i < algo.fbest.size()){
                file << algo.fbest[i] << ",";
            } else {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

void exportStatisticsToText(const vector<AlgorithmResults>& algorithms, const string& outputFile) {
    ofstream file(outputFile);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << outputFile << endl;
        return;
    }

    file << "---------------------------------------------------------------------------------------------";

    file << " " << endl;

    file << " Quantity       Statistic  ";
    for (const auto& algo : algorithms) {
        file << "" << setw(13) << algo.name;
    }
    file << "" << endl;

    file << "---------------------------------------------------------------------------------------------";

    file << "" << endl;

    file << "   fbest        Mean       ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.fbest, mean, median, stdev, min, max);
        file << "" << setw(13) << mean;
    }
    file << " " << endl;

    file << "                Median     ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.fbest, mean, median, stdev, min, max);
        file << "" << setw(13) << median;
    }
    file << " " << endl;

    file << "                St.dev.    ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.fbest, mean, median, stdev, min, max);
        file << "" << setw(13) << stdev;
    }
    file << " " << endl;

    file << "                Min        ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.fbest, mean, median, stdev, min, max);
        file << "" << setw(13) << min;
    }
    file << " " << endl;

    file << "                Max        ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.fbest, mean, median, stdev, min, max);
        file << "" << setw(13) << max;
    }
    file << " " << endl;
    file << "---------------------------------------------------------------------------------------------" << endl;

    file << "   last-hit     Mean       ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.last_hit, mean, median, stdev, min, max);  
        file << "" << setw(13) << mean;
    }
    file << " " << endl;

    file << "                Median     ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.last_hit, mean, median, stdev, min, max);  
        file << "" << setw(13) << median;
    }
    file << " " << endl;

    file << "                St.dev.    ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.last_hit, mean, median, stdev, min, max); 
        file << "" << setw(13) << stdev;
    }
    file << " " << endl;

    file << "                Min        ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.last_hit, mean, median, stdev, min, max);  
        file << "" << setw(13) << min;
    }
    file << " " << endl;

    file << "                Max        ";
    for (const auto& algo : algorithms) {
        double mean, median, stdev, min, max;
        computeStatistics(algo.last_hit, mean, median, stdev, min, max);  
        file << "" << setw(13) << max;
    }
    file << " " << endl;

    file << "---------------------------------------------------------------------------------------------";

    file.close();
}

double computeWilcoxonPValue(const vector<double>& sample1, const vector<double>& sample2) {
    size_t n1 = sample1.size();
    size_t n2 = sample2.size();
    size_t N = n1 + n2;

    vector<pair<double, size_t>> values;
    values.reserve(N);
    for (size_t i = 0; i < n1; ++i) {
        values.emplace_back(sample1[i], 0);
    }
    for (size_t i = 0; i < n2; ++i) {
        values.emplace_back(sample2[i], 1);
    }

    sort(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    vector<double> ranks(N, 0.0);
    size_t i = 0;
    double tie_correction = 0.0;
    while (i < N) {
        size_t j = i + 1;
        while (j < N && values[j].first == values[i].first) {
            ++j;
        }
        double rank_avg = (i + 1 + j) / 2.0;
        for (size_t k = i; k < j; ++k) {
            ranks[k] = rank_avg;
        }
        size_t t = j - i;
        if (t > 1) {
            tie_correction += static_cast<double>(t * t * t - t);
        }
        i = j;
    }

    double rankSum1 = 0.0;
    for (size_t idx = 0; idx < N; ++idx) {
        if (values[idx].second == 0) {
            rankSum1 += ranks[idx];
        }
    }

    double U1 = rankSum1 - (n1 * (n1 + 1)) / 2.0;
    double U2 = (n1 * n2) - U1;
    double U = min(U1, U2);

    double meanU = (n1 * n2) / 2.0;
    double varU = (n1 * n2) / 12.0 * (static_cast<double>(N + 1) - tie_correction / (static_cast<double>(N) * (N - 1)));
    double stddevU = sqrt(varU);
    if (stddevU == 0.0) {
        return 1.0;
    }

    double z = (U - meanU + 0.5) / stddevU;
    double cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)));
    double pValue = 2.0 * min(cdf, 1.0 - cdf);

    return pValue;
}
void generateSignificanceTable(const vector<string>& algorithms, const vector<vector<double>>& fbestSamples, const string& outputFile) {
    ofstream file(outputFile);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << outputFile << endl;
        return;
    }

    file << "----------------------------------------------------------------------------------------------------------" << endl;
    file << "               ";
    for (const auto& algo : algorithms) {
        file << setw(18) << algo;
    }
    file << endl;

    file << "----------------------------------------------------------------------------------------------------------" << endl;

    for (size_t i = 0; i < algorithms.size(); ++i) {
        file << setw(15) << algorithms[i];

        for (size_t j = 0; j < algorithms.size(); ++j) {
            if (i == j) {
                file << setw(15) << "-"; 
            } else {
                double pValue = computeWilcoxonPValue(fbestSamples[i], fbestSamples[j]);

                string symbol;
                if (pValue > 0.05) {
                    symbol = "≈";
                } else if (pValue < 0.05) {
                    double mean_i = accumulate(fbestSamples[i].begin(), fbestSamples[i].end(), 0.0) / fbestSamples[i].size();
                    double mean_j = accumulate(fbestSamples[j].begin(), fbestSamples[j].end(), 0.0) / fbestSamples[j].size();
                    symbol = (mean_i < mean_j) ? "+" : "-";  
                }

                file << setw(15) << pValue << " (" << symbol << ")";
            }
        }
        file << endl;
    }

    file << "----------------------------------------------------------------------------------------------------------" << endl;

    file.close();
}
