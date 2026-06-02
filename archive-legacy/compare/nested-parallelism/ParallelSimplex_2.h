#pragma once

#include <oneapi/tbb/task_arena.h>

#include "SharedData.h"

enum class SimplexStatus {
    OPTIMAL,      // знайдено оптимальний розв'язок
    UNBOUNDED,    // задача необмежена
    INFEASIBLE,   // задача недопустима (після preprocess з b[i] < 0)
    MAX_ITER      // досягнуто максимальну кількість ітерацій
};

class ParallelSimplex {
    public:
        ParallelSimplex(SharedData& data,
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
        SharedData& data;
        int P;
        int maxIter;
        mutable tbb::task_arena arena;
        uint32_t seed;

        mutable std::vector<tbb::task_arena> inner_arenas;

        alignas(64) std::vector<double> v;      // розмір 1 x m

        int j_pivot{-1};
        int i_pivot{-1};

        double result{0.0};
        int iterations{0};
        std::vector<double> equationSolution;

        void computeVectorV();

        template<typename InnerFn>
        int computeJPivot(std::pair<double, int> identity, InnerFn&& innerFn);

        void computeVectorD(std::vector<double>& d);
        void performBasisUpdate(const std::vector<double>& d, int ip, int jp);
        int harrisRatioPivot(const std::vector<double>& d) const;
        int computeIPivotForDual() const;

        void restoreCTB();
};
