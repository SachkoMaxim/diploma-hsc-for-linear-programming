#include "SharedData.h"

#include <random>
#include <numeric>

void SharedData::initialise(int n_vars, int m_constraints) {
    n = n_vars;
    m = m_constraints;

    d_ipRev = 0.0;
    x_B_iPiv = 0.0;
    needsDualStart = false;

    // Виділення пам'яті під вектори
    b.resize(m, 0.0);               // розмір m
    c.resize(n, 0.0);               // розмір n
    c_TB.resize(m, 0.0);            // розмір m
    x_B.resize(m, 0.0);             // розмір m
    B_m1_iPiv.resize(m, 0.0);       // розмір m

    // Виділення пам'яті під одновимірні матриці
    A_T.resize(n * m, 0.0);     // розмір n x m
    B.resize(m * m, 0.0);       // розмір m x m
    B_m1.resize(m * m, 0.0);    // розмір m x m

    basisIdx.resize(m);
    std::iota(basisIdx.begin(), basisIdx.end(), n);

    constraintTypes.resize(m, ConstraintType::LESSEREQ);
    constraintTypes[1] = ConstraintType::GREATEREQ;
}

void SharedData::putValuesIntoVector(
    int size, std::vector<double> &vector, double min, double max, uint32_t seed
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(static_cast<int>(min), static_cast<int>(max));

    for (int i = 0; i < size; ++i) {
        vector[i] = static_cast<double>(dist(rng));
    }
}

void SharedData::putValuesIntoMatrix(
    int rowSize, int colSize, std::vector<double> &matrix, double min, double max, uint32_t seed
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(static_cast<int>(min), static_cast<int>(max));

    // Матриця має загальний розмір rowSize * colSize
    int totalElements = rowSize * colSize;

    for (int i = 0; i < totalElements; ++i) {
        matrix[i] = static_cast<double>(dist(rng));
    }
}

void SharedData::putValuesIntoIdentityMatrix(int rowSize, int colSize, std::vector<double> &matrix) {
    if (rowSize != colSize) return;

    std::fill(matrix.begin(), matrix.end(), 0.0);

    int step = rowSize + 1;
    int totalElements = rowSize * colSize;

    for (int i = 0; i < totalElements; i += step) matrix[i] = 1.0;
}

void SharedData::preprocessA_T() {
    for (int i = 0; i < m; ++i) {
        if (constraintTypes[i] == ConstraintType::GREATEREQ) {
            for (int j = 0; j < n; ++j) A_T[j * m + i] = -A_T[j * m + i];
        }
    }
}

void SharedData::preprocessB() {
    for (int i = 0; i < m; ++i) {
        if (constraintTypes[i] == ConstraintType::GREATEREQ) {
            b[i] = -b[i];
        }
    }
}

void SharedData::putValuesIntoA_T(std::vector<double> &A, double min, double max, uint32_t seed) {
    putValuesIntoMatrix(n, m, A, min, max, seed);
    preprocessA_T();
}

void SharedData::putValuesIntoB(std::vector<double> &b, double min, double max, uint32_t seed) {
    putValuesIntoVector(m, b, min, max, seed);
    preprocessB();
}

void SharedData::saveScalar(double &scalar, std::vector<double> &vector, int i_pivot) {
    scalar = vector[i_pivot];
}

void SharedData::saveRow(std::vector<double> &vector_copy, int i_pivot) {
    const double* sourceRow = rowB_m1(i_pivot);

    std::copy(sourceRow, sourceRow + m, vector_copy.begin());
}

void SharedData::changeElement(std::vector<double> &vector, int i_pivot, int j_pivot) {
    int ip = i_pivot;
    int jp = j_pivot;
    // Вхідна змінна jp входить у базис замість вихідної ip
    basisIdx[ip] = jp;
    vector[ip] = (jp < n) ? c[jp] : 0.0;
}
