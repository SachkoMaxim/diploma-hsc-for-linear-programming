#include <iostream>
#include <sstream>
#include <chrono>

#include "src/FileWriter.h"
#include "src/ConsoleVisualizer.h"
#include "src/ParallelSimplex.h"
#include "src/SharedData.h"

const int n = 5;
const int m = 4;
const int P = 4;

const uint32_t SEED = 77;

using namespace std;

int main() {

    const int MAX_ITER = 100000;

    std::cout << "Program has started\n\n";

    SharedData data;
    data.initialise(n, m);

    ParallelSimplex solver(data, P, MAX_ITER, SEED);

    auto startTime = std::chrono::high_resolution_clock::now();
    SimplexStatus status = solver.solve();

    // Виведення часу виконання
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    cout << "\nProblem condition:\n";
    std::cout << "n = " << n << ", m = " << m << ", P = " << P
                      << ", (iter=" << solver.getIterationCount() << ")"
                      << std::endl;

    bool is_large_problem = (n > 15 || m > 15);

    if (!is_large_problem) {
        ConsoleVisualizer::printVector("Vector b", data.b);
        ConsoleVisualizer::printVector("Vector c", data.c);
        ConsoleVisualizer::printMatrixTransposed("Matrix A", data.A_T, data.n, data.m);

        cout << "\nResults:\n";
        ConsoleVisualizer::printSimplexStatus("Simplex Algorithm status", status);

        if (status == SimplexStatus::OPTIMAL) {
            ConsoleVisualizer::printVector("x", solver.getEquationSolution());
            std::cout << "Z = " << solver.getResult() << std::endl;
        } else {
            std::cout << "[INFO] The final solution vector 'x' is missing due to the suboptimal status of the algorithm.\n";
        }
    } else {
        cout << "\nResults (Large Problem Mode):\n";
        ConsoleVisualizer::printSimplexStatus("Simplex Algorithm status", status);
        if (status == SimplexStatus::OPTIMAL) {
            std::cout << "Z = " << solver.getResult() << std::endl;
        }

        // Виклик функції запису у файл для великих даних
        std::cout << "[FILE LOG] Writing data in file simplex_results.txt...\n";
        FileWriter::saveLargeResultToFile(
            "simplex_results.txt", status, solver.getEquationSolution(), solver.getResult(),
            solver.getIterationCount(), n, m, P, data.b, data.c, data.A_T
        );
    }

    std::cout << "\nThe program execution time was " << (double)duration.count() / 1000.0F << " seconds" << std::endl;

    return 0;
}
