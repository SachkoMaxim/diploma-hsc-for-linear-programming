#pragma once

#include "SharedData.h"

enum class SimplexStatus {
    OPTIMAL,      // знайдено оптимальний розв'язок
    UNBOUNDED,    // задача необмежена
    INFEASIBLE,   // задача недопустима (після preprocess з b[i] < 0)
    MAX_ITER      // досягнуто максимальну кількість ітерацій
};

class ParallelSimplex {
    public:
        ParallelSimplex(SharedData &data,
            int numThreads = 4,
            int maxIter = 100000,
            uint32_t seed = 1
        );

        ~ParallelSimplex() = default;

        SimplexStatus solve();

        double getResult() const { return result; }
        int getIterationCount() const { return iterations; }
        const std::vector<double>& getEquationSolution() const { return equationSolution; }

    private:
        SharedData &data;
        int P;
        int maxIter;

        uint32_t seed;

        alignas(64) std::vector<double> v;      // розмір 1 x m

        int j_pivot{-1};
        int i_pivot{-1};

        double result = 0.0;
        int iterations = 0;
        std::vector<double> equationSolution;

        void computeReducedCosts(std::vector<double> &c_j);
        void performBasisUpdate(const std::vector<double> &d, int ip, int jp);
        int harrisRatioPivot(const std::vector<double> &d) const;
};
