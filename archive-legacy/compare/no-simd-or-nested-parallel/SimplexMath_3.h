#pragma once

#include <vector>
#include <oneapi/tbb.h>

class SimplexMath {
    public:
        static inline void MultiplyRowAndTransposedMatrix(
            const std::vector<double>& row, const std::vector<double>& matrix_T,
            std::vector<double>& result, int rowSize, int resultSize
        ) {
            tbb::parallel_for(tbb::blocked_range<int>(0, resultSize), [&](const tbb::blocked_range<int>& r) {
                for (int j = r.begin(); j < r.end(); ++j) {
                    double sum = 0.0;
                    const double* col = matrix_T.data() + j * rowSize;
                    // Звичайний послідовний цикл без SIMD
                    for (int i = 0; i < rowSize; ++i)
                        sum += row[i] * col[i];
                    result[j] = sum;
                }
            });
        }

        static inline void UpdateMatrixAndVectorParallel(
            const std::vector<double>& piv_row, const std::vector<double>& d, const double scalar,
            std::vector<double>& matrix, const double x_B_piv, std::vector<double>& x_B,
            int rows, int cols, int i_pivot
        ) {
            tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i < r.end(); ++i) {
                    double* row_ptr = matrix.data() + i * cols;

                    if (i == i_pivot) {
                        // Без SIMD
                        for (int j = 0; j < cols; ++j)
                            row_ptr[j] = scalar * piv_row[j];
                        x_B[i] = scalar * x_B_piv;
                    } else {
                        const double f_i = scalar * d[i];
                        if (f_i == 0.0) continue;
                        // Без SIMD
                        for (int j = 0; j < cols; ++j)
                            row_ptr[j] -= f_i * piv_row[j];
                        x_B[i] -= f_i * x_B_piv;
                    }
                }
            });
        }
};
