#pragma once

#include <vector>
#include <oneapi/tbb.h>
#include <functional>

class SimplexMath {
    public:
        static inline void MultiplyRowAndTransposedMatrix(
            const std::vector<double>& row, const std::vector<double>& matrix_T,
            std::vector<double>& result, int rowSize, int resultSize,
            std::vector<tbb::task_arena>& inner_arenas
        ) {
            tbb::parallel_for(tbb::blocked_range<int>(0, resultSize), [&](const tbb::blocked_range<int>& r) {

                int th_idx = tbb::this_task_arena::current_thread_index();

                for (int j = r.begin(); j < r.end(); ++j) {
                    const double* col = matrix_T.data() + j * rowSize;
                    double sum = 0.0;

                    inner_arenas[th_idx].execute([&]() {
                        sum = tbb::parallel_reduce(
                            tbb::blocked_range<int>(0, rowSize),
                            0.0,
                            [&](const tbb::blocked_range<int>& inner_r, double local_sum) {
                                for (int i = inner_r.begin(); i < inner_r.end(); ++i)
                                    local_sum += row[i] * col[i];
                                return local_sum;
                            },
                            std::plus<double>()
                        );
                    });

                    result[j] = sum;
                }
            });
        }

        static inline void UpdateMatrixAndVectorParallel(
            const std::vector<double>& piv_row, const std::vector<double>& d, const double scalar,
            std::vector<double>& matrix, const double x_B_piv, std::vector<double>& x_B,
            int rows, int cols, int i_pivot, std::vector<tbb::task_arena>& inner_arenas
        ) {
            tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& r) {

                int th_idx = tbb::this_task_arena::current_thread_index();

                for (int i = r.begin(); i < r.end(); ++i) {
                    double* row_ptr = matrix.data() + i * cols;

                    if (i == i_pivot) {
                        inner_arenas[th_idx].execute([&]() {
                            tbb::parallel_for(tbb::blocked_range<int>(0, cols), [&](const tbb::blocked_range<int>& inner_r) {
                                for (int j = inner_r.begin(); j < inner_r.end(); ++j)
                                    row_ptr[j] = scalar * piv_row[j];
                            });
                        });
                        x_B[i] = scalar * x_B_piv;
                    } else {
                        const double f_i = scalar * d[i];
                        if (f_i == 0.0) continue;

                        inner_arenas[th_idx].execute([&]() {
                            tbb::parallel_for(tbb::blocked_range<int>(0, cols), [&](const tbb::blocked_range<int>& inner_r) {
                                for (int j = inner_r.begin(); j < inner_r.end(); ++j)
                                    row_ptr[j] -= f_i * piv_row[j];
                            });
                        });
                        x_B[i] -= f_i * x_B_piv;
                    }
                }
            });
        }
};
