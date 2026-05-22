class SimplexMath {
    inline void MultiplyScalarAndVector(const double scalar, const double* vector, double* result, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] = scalar * vector[i];
        }
    }

    inline void DivideVectors(const double* vector1, const double* vector2, double* result, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] = vector1[i] / vector2[i];
        }
    }

    inline void MultiplyVectorAndMatrix(
        const double* vector, const double* matrix, double* result, int col, int row
    ) {
        for (int j = 0; j < row; ++j) result[j] = 0.0;

        for (int i = 0; i < col; ++i) {
            // Знаходження потрібного рядка
            const double* row_ptr = matrix + i * row;
            double vec_val = vector[i];

            for (int j = 0; j < row; ++j) {
                result[j] += vec_val * row_ptr[j];
            }
        }
    }

    inline void MultiplyMatrixAndColumn(
        const double* matrix, const double* A, int j_pivot, double* result, int m, int n
    ) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            // Знаходження потрібного рядка
            const double* row_ptr = matrix + i * m;
            for (int j = 0; j < m; ++j) {
                // A[j * n + j_pivot] - елемент у рядку j та стовпці j_pivot матриці A
                sum += row_ptr[j] * A[j * n + j_pivot];
            }
            result[i] = sum;
        }
    }

    inline void UpdateVector(const double* d, const double piv_el, const double scalar, double* result, int size, int i_pivot) {;
        for (int i = 0; i < size; ++i) {
            if (i == i_pivot) {
                result[i] = scalar * piv_el;
            } else {
                double f_i = scalar * d[i];
                result[i] -= f_i * piv_el;
            }
        }
    }

    inline void UpdateMatrix(const double* piv_row, const double* d, const double scalar, double* matrix, int row, int col, int i_pivot) {
        for (int i = 0; i < row; ++i) {
            double* row_ptr = matrix + i * col;
            if (i == i_pivot) {
                for (int j = 0; j < col; ++j) {
                    row_ptr[j] = scalar * piv_row[j];
                }
            } else {
                double f_i = scalar * d[i];
                if (f_i == 0.0) continue;
                for (int j = 0; j < col; ++j) {
                    row_ptr[j] -= f_i * piv_row[j];
                }
            }
        }
    }
};
