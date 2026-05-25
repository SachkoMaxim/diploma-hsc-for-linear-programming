#pragma once

#include <vector>

class SimplexMath {
    public:
        static inline void MultiplyScalarAndVector(
            const double scalar, const std::vector<double> &vector, std::vector<double> &result, int size
        ) {
            for (int i = 0; i < size; ++i) {
                result[i] = scalar * vector[i];
            }
        }

        static inline void DivideVectors(
            const std::vector<double> &vector1, const std::vector<double> &vector2,
            std::vector<double> &result, int size
        ) {
            for (int i = 0; i < size; ++i) {
                if (vector2[i] > 1e-9) {
                    result[i] = vector1[i] / vector2[i];
                }
            }
        }

        static inline void MultiplyVectorAndMatrix(
            const std::vector<double> &vector, const std::vector<double> &matrix,
            std::vector<double> &result, int vectorSize, int resultSize
        ) {
            for (int i = 0; i < vectorSize; ++i) {
                double sum = 0.0;
                double vec_val = vector[i];

                for (int j = 0; j < resultSize; ++j) {
                    result[j] += vec_val * matrix[i * resultSize + j];
                }
            }
        }

        static inline void MultiplyVectorAndTransposedMatrix(
            const std::vector<double> &vector, const std::vector<double> &matrix_T,
            std::vector<double> &result, int vectorSize, int resultSize
        ) {
            for (int j = 0; j < resultSize; ++j) {
                double sum = 0.0;
                const double* row_mT = matrix_T.data() + j * vectorSize;

                for (int i = 0; i < vectorSize; ++i) sum += vector[i] * row_mT[i];

                result[j] = sum;
            }
        }

        static inline void MultiplyMatrixAndColumn(
            const std::vector<double> &matrix, const std::vector<double> &A_T, int j_pivot,
            std::vector<double> &result, int m
        ) {
            const double* col = A_T.data() + j_pivot * m;

            for (int i = 0; i < m; ++i) {
                double sum = 0.0;
                const double* row = matrix.data() + i * m;

                for (int j = 0; j < m; ++j) sum += row[j] * col[j];
                result[i] = sum;
            }
        }

        static inline void UpdateVector(
            const std::vector<double> &d, const double piv_el, const double scalar,
            std::vector<double> &result, int size, int i_pivot
        ) {
            for (int i = 0; i < size; ++i) {
                if (i == i_pivot) {
                    result[i] = scalar * piv_el;
                } else {
                    double f_i = scalar * d[i];
                    result[i] = result[i] - f_i * piv_el;
                }
            }
        }

        static inline void UpdateMatrix(
            const std::vector<double> &piv_row, const std::vector<double> &d,
            const double scalar, std::vector<double> &matrix, int row, int col, int i_pivot
        ) {
            for (int i = 0; i < row; ++i) {
                int row_offset = i * col;

                if (i == i_pivot) {
                    for (int j = 0; j < col; ++j) {
                        matrix[row_offset + j] = scalar * piv_row[j];
                    }
                } else {
                    double f_i = scalar * d[i];
                    if (f_i == 0.0) continue;
                    for (int j = 0; j < col; ++j) {
                        matrix[row_offset + j] -= f_i * piv_row[j];
                    }
                }
            }
        }

        static inline void UpdateMatrixAndVectorParallel(
            const std::vector<double> &piv_row, const std::vector<double> &d, const double scalar,
            std::vector<double> &matrix, const double x_B_piv, std::vector<double> &x_B,
            int row, int col, int i_pivot
        ) {
#pragma omp parallel for
            for (int i = 0; i < row; ++i) {
                int row_offset = i * col;

                if (i == i_pivot) {
                    // Обчислення11 (B_m1)н = d_ipRev * B_m1_iPiv
                    for (int j = 0; j < col; ++j) {
                        matrix[row_offset + j] = scalar * piv_row[j];
                    }

                    // Обчислення12 (x_B)н = d_ipRev * x_B_piv
                    x_B[i] = scalar * x_B_piv;
                }
                else {
                    // Обчислення10 fн = d_ipRev * dн
                    double f_i = scalar * d[i];

                    // Якщо коефіцієнт нульовий, цей потік пропускає і матрицю, і вектор для даного i
                    if (f_i == 0.0) continue;

                    // Обчислення11 (B_m1)н = (B_m1)н - fн * B_m1_iPiv
                    for (int j = 0; j < col; ++j) {
                        matrix[row_offset + j] -= f_i * piv_row[j];
                    }

                    // Обчислення12 (x_B)н = (x_B)н - fн * x_B_piv
                    x_B[i] -= f_i * x_B_piv;
                }
            } // Фінальний бар'єр B2: і матриця, і вектор готові одночасно!
        }
};
