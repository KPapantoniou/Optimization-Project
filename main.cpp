#include <iostream>
#include <vector>
#include <fstream>
#include <array>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "utils.h"
#include "nelderMead.h"
#include "newtonTr.h"
#include "optimization.h"
#include "bfgs.h"
#include "ga.h"
#include "pso.h"

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    initializeLogFile("initial_points.txt");
    initializeLogFile("output_NelderMead_train.txt");
    initializeLogFile("output_NewtonTR_train.txt");
    initializeLogFile("output_BFGSWolfe_train.txt");
    initializeLogFile("output_GA_train.txt");
    initializeLogFile("output_PSO_train.txt");
    initializeLogFile("output_test.txt");
    initializeLogFile("statistics.txt");
    initializeLogFile("results.csv");
    initializeLogFile("significance.txt");


    auto afterLogInit = std::chrono::high_resolution_clock::now();
    std::cout << "Log files initialized in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(afterLogInit - start).count() 
              << " ms" << std::endl;

    generateAndSaveInitialPoints(parameterRanges, "initial_points.txt");

    auto afterGeneratePoints = std::chrono::high_resolution_clock::now();
    std::cout << "Initial points generated in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(afterGeneratePoints - afterLogInit).count() 
              << " ms" << std::endl;

    auto data_train = readData("data_train.txt");
    auto initialPoints = loadInitialPoints("initial_points.txt");
    auto data_test = readData("data_test.txt");

    auto afterDataLoad = std::chrono::high_resolution_clock::now();
    std::cout << "Data loaded in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(afterDataLoad - afterGeneratePoints).count() 
              << " ms" << std::endl;

    
    
    std::vector<double> mseTestNM;
    std::vector<double> mseTestNT;
    std::vector<double> mseTestBFGS;
    std::vector<double> mseTestGA;
    std::vector<double> mseTestPSO;

    auto algoStart = std::chrono::high_resolution_clock::now();

    std::thread tNewtonTr([&] {
        NewtonTr ntr(1e5, 1e-7);
        mseTestNT = runExperiment(ntr, data_train, data_test, initialPoints, "output_NewtonTR_train.txt");
    });
    std::thread tNelderMead([&] {
        NelderMead nm(1e5, 1e-7);
        mseTestNM = runExperiment(nm, data_train, data_test, initialPoints, "output_NelderMead_train.txt");
    });
    std::thread tGA([&] {
        GA ga(1e5, 1e-7);
        mseTestGA = runExperiment(ga, data_train, data_test, initialPoints, "output_GA_train.txt");
    });
    std::thread tPSO([&] {
        PSO pso(1e5, 1e-7);
        mseTestPSO = runExperiment(pso, data_train, data_test, initialPoints, "output_PSO_train.txt");
    });
    std::thread tBFGS([&] {
        BFGS bfgs(1e5, 1e-7);
        mseTestBFGS = runExperiment(bfgs, data_train, data_test, initialPoints, "output_BFGSWolfe_train.txt");
    });

    tNewtonTr.join();
    tNelderMead.join();
    tGA.join();
    tPSO.join();
    tBFGS.join();

    auto algoEnd = std::chrono::high_resolution_clock::now();
    std::cout << "All optimization algorithms completed in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(algoEnd - algoStart).count()
              << " ms" << std::endl;


    std::vector<double> NelderMeadResults;
    std::vector<double> NewtonTrResults;
    std::vector<double> BFGSResults;
    std::vector<double> GAResults;
    std::vector<double> PSOResults;

    readResults("output_NelderMead_train.txt", NelderMeadResults, 1);
    readResults("output_NewtonTR_train.txt", NewtonTrResults, 1);
    readResults("output_BFGSWolfe_train.txt", BFGSResults, 1);
    readResults("output_GA_train.txt", GAResults, 1);
    readResults("output_PSO_train.txt", PSOResults, 1);

    std::vector<double> NelderMeadLastHit;
    std::vector<double> NewtonTrLastHit;
    std::vector<double> BFGSLastHit;
    std::vector<double> GALastHit;
    std::vector<double> PSOLastHit;

    readResults("output_NelderMead_train.txt", NelderMeadLastHit, 2);
    readResults("output_NewtonTR_train.txt", NewtonTrLastHit, 2);
    readResults("output_BFGSWolfe_train.txt", BFGSLastHit, 2);
    readResults("output_GA_train.txt", GALastHit, 2);
    readResults("output_PSO_train.txt", PSOLastHit, 2);

    std::vector<AlgorithmResults> results;
    results.push_back({"NelderMead", NelderMeadResults, NelderMeadLastHit});
    results.push_back({"NewtonTR", NewtonTrResults, NewtonTrLastHit});
    results.push_back({"BFGSWolfe", BFGSResults, BFGSLastHit});
    results.push_back({"GA", GAResults, GALastHit});
    results.push_back({"PSO", PSOResults, PSOLastHit});


    exportToCSV(results, "results.csv");

    exportStatisticsToText(results, "statistics.txt");

    generateSignificanceTable({"NelderMead", "NewtonTR", "BFGSWolfe", "GA", "PSO"}, {NelderMeadResults, NewtonTrResults, BFGSResults, GAResults, PSOResults}, "significance.txt");


    std::vector<std::vector<double>> mseResults(30, std::vector<double>(5));
    for (size_t i = 0; i < 30; ++i) {
    mseResults[i][0] = mseTestNT[i];
    mseResults[i][1] = mseTestBFGS[i];
    mseResults[i][2] = mseTestNM[i];
    mseResults[i][3] = mseTestGA[i];
    mseResults[i][4] = mseTestPSO[i];
}
    writeTestResultsToFile("output_test.txt", mseResults);

    std::string command = "python3 boxplot.py";
    int reternVal = system(command.c_str());

    if (reternVal == 0) {
        std::cout << "Boxplot generated successfully." << std::endl;
    } else {
        std::cerr << "Error generating boxplot." << std::endl;
    }
    
    
    std::cout << "All optimization algorithms completed." << std::endl;
    return 0;
}




    

