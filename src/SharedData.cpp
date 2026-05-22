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
    A.resize(m * n, 0.0);           // розмір m x n
    B.resize(m * m, 0.0);           // розмір m x m
    B_m1.resize(m * m, 0.0);        // розмір m x m

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

void SharedData::preprocess() {
    for (int i = 0; i < m; ++i) {
        if (constraintTypes[i] == ConstraintType::GREATEREQ) {
            double* row = rowA(i);
            for (int j = 0; j < n; ++j) row[j] = -row[j];
            b[i] = -b[i];
            constraintTypes[i] = ConstraintType::LESSEREQ;
            if (b[i] < 0.0) needsDualStart = true;
        }
    }
}

void SharedData::putValuesIntoA(std::vector<double> &A, double min, double max, uint32_t seed) {
    putValuesIntoMatrix(m, n, A, min, max, seed);
    preprocess();
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
