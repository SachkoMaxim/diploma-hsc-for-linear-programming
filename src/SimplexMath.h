#pragma once

#include <vector>

class SimplexMath {
    public:
        static inline void MultiplyRowAndTransposedMatrix(
            const std::vector<double>& row, const std::vector<double>& matrix_T,
            std::vector<double>& result, int rowSize, int resultSize, int P
        ) {
#pragma omp parallel for schedule(static) num_threads(P)
            for (int j = 0; j < resultSize; ++j) {
                double sum = 0.0;
                const double* col = matrix_T.data() + j * rowSize;
#pragma omp simd reduction(+:sum)
                for (int i = 0; i < rowSize; ++i)
                    sum += row[i] * col[i];
                result[j] = sum;
            }
        }

        static inline void UpdateMatrixAndVectorParallel(
            const std::vector<double>& piv_row, const std::vector<double>& d, const double scalar,
            std::vector<double>& matrix, const double x_B_piv, std::vector<double>& x_B,
            int rows, int cols, int i_pivot, int P
        ) {
#pragma omp parallel for schedule(static) num_threads(P)
            for (int i = 0; i < rows; ++i) {
                double* row_ptr = matrix.data() + i * cols;

                if (i == i_pivot) {
#pragma omp simd
                    // Обчислення11 (B_m1)н = d_ipRev * B_m1_iPiv
                    for (int j = 0; j < cols; ++j)
                        row_ptr[j] = scalar * piv_row[j];
                    // Обчислення12 (x_B)н = d_ipRev * x_B_piv
                    x_B[i] = scalar * x_B_piv;
                } else {
                    // Обчислення10 fн = d_ipRev * dн
                    const double f_i = scalar * d[i];
                    if (f_i == 0.0) continue; // рядок не змінюється
#pragma omp simd
                    // Обчислення11 (B_m1)н = (B_m1)н - fн * B_m1_iPiv
                    for (int j = 0; j < cols; ++j)
                        row_ptr[j] -= f_i * piv_row[j];
                    // Обчислення12 (x_B)н = (x_B)н - fн * x_B_piv
                    x_B[i] -= f_i * x_B_piv;
                }
            } // Бар'єр B3: матриця і вектор готові одночасно
        }
};
