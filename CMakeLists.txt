cmake_minimum_required(VERSION 3.16)

project(OptimizationBonusProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_SOURCES
    main.cpp
    utils.cpp
    optimization.cpp
    nelderMead.cpp
    newtonTr.cpp
    bfgs.cpp
    ga.cpp
    pso.cpp
)

add_executable(optimization_project ${PROJECT_SOURCES})
target_include_directories(optimization_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(RUNTIME_INPUTS
    data_train.txt
    data_test.txt
    boxplot.py
)

foreach(input_file IN LISTS RUNTIME_INPUTS)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${input_file}
        ${CMAKE_CURRENT_BINARY_DIR}/${input_file}
        COPYONLY
    )
endforeach()

set(GENERATED_ARTIFACTS
    initial_points.txt
    nelder_mead_results.txt
    newton_tr_results.txt
    BFGS_results.txt
    GA_results.txt
    pso_results.txt
    test_results.txt
    test_result_NM.txt
    test_result_NT.txt
    test_result_BFGS.txt
    test_result_GA.txt
    test_result_PSO.txt
    statistics.txt
    significance.txt
    results.csv
    output_test.csv
    output_test.txt
    solution_boxplot.png
)

add_custom_target(clean_generated
    COMMAND ${CMAKE_COMMAND} -E rm -f ${GENERATED_ARTIFACTS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Removing previously generated output artifacts"
)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E rm -f ${GENERATED_ARTIFACTS}
    COMMAND $<TARGET_FILE:optimization_project>
    DEPENDS optimization_project
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Building and running optimization_project"
)
