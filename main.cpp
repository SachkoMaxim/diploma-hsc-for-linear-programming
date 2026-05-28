#include <iostream>
#include <chrono>

#include "src/FileWriter.h"
#include "src/ConsoleVisualizer.h"
#include "src/ParallelSimplex.h"
#include "src/SharedData.h"

static constexpr int n = 5;
static constexpr int m = 4;
static constexpr int P = 4;
static constexpr uint32_t SEED = 77;
static constexpr int MAX_ITER = 100000;

int main() {
    std::cout << "Program started\n\n";

    SharedData data;
    data.initialise(n, m);

    ParallelSimplex solver(data, P, MAX_ITER, SEED);

    const auto startTime = std::chrono::high_resolution_clock::now();
    SimplexStatus status = solver.solve();

    // Виведення часу виконання
    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\nProblem: n=" << n << ", m=" << m << ", P=" << P
              << ", iterations=" << solver.getIterationCount() << "\n";

    const bool isLarge = (n > 15 || m > 15);

    if (!isLarge) {
        ConsoleVisualizer::printVector("b", data.b);
        ConsoleVisualizer::printVector("c", data.c);
        ConsoleVisualizer::printMatrixTransposed("A", data.A_T, data.n, data.m);
        std::cout << "\nResults:\n";
        ConsoleVisualizer::printSimplexStatus("Status", status);
        if (status == SimplexStatus::OPTIMAL) {
            ConsoleVisualizer::printVector("x", solver.getEquationSolution());
            std::cout << "Z = " << solver.getResult() << "\n";
        } else {
            std::cout << "[INFO] Solution vector unavailable (non-optimal status).\n";
        }
    } else {
        std::cout << "\nResults (large problem):\n";
        ConsoleVisualizer::printSimplexStatus("Status", status);
        if (status == SimplexStatus::OPTIMAL)
            std::cout << "Z = " << solver.getResult() << "\n";

        std::cout << "[FILE LOG] Writing results to simplex_results.txt ...\n";
        FileWriter::saveLargeResultToFile(
            "simplex_results.txt", status, solver.getEquationSolution(), solver.getResult(),
            solver.getIterationCount(), n, m, P, data.b, data.c, data.A_T
        );
    }

    std::cout << "\nThe program execution time was " << (double)duration.count() / 1000.0F << " seconds" << std::endl;

    return 0;
}
