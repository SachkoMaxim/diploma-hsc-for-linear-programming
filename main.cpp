#include <iostream>

#include "ConsoleVisualizer.h"
#include "src/SimplexMethod.h"
#include "src/SharedData.h"

const int n = 3;
const int m = 3;
const int P = 4;

const uint32_t SEED = 1;

using namespace std;

int main() {

    const int MAX_ITER = 100000;

    SharedData data;
    data.initialise(n, m);

    SimplexMethod solver(data, P, MAX_ITER, SEED);
    SimplexStatus status = solver.solve();

    std::cout << "n = " << n << ", m = " << m << ", P = " << P
                      << ", (iter=" << solver.getIterationCount() << ")"
                      << std::endl;

    ConsoleVisualizer::printVector("x",solver.getEquationSolution());

    std::cout << "Z = " << solver.getResult() << std::endl;

    const auto lang = "C++";
    std::cout << "Hello and welcome to " << lang << "!\n";

    return 0;
}
