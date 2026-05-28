#include "SharedData.h"

#include <random>
#include <numeric>
#include <algorithm>

void SharedData::initialise(int n_vars, int m_constraints) {
    n = n_vars;
    m = m_constraints;

    d_ipRev.value  = 0.0;
    x_B_iPiv.value = 0.0;
    needsDualStart = false;

    // Вектори
    b.assign(m, 0.0);
    c.assign(n + m, 0.0);
    c_TB.assign(m, 0.0);
    x_B.assign(m, 0.0);
    B_m1_iPiv.assign(m, 0.0);

    // Матриці
    A_T.assign(static_cast<size_t>(n + m) * m, 0.0);
    B.assign(static_cast<size_t>(m) * m, 0.0);
    B_m1.assign(static_cast<size_t>(m) * m, 0.0);

    basisIdx.resize(m);
    std::iota(basisIdx.begin(), basisIdx.end(), n);

    constraintTypes.assign(m, ConstraintType::LESSEREQ);
    constraintTypes[1] = ConstraintType::GREATEREQ;
}

void SharedData::putValuesIntoVector(
    int size, std::vector<double>& vec, double min, double max, uint32_t seed
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(static_cast<int>(min), static_cast<int>(max));

    for (int i = 0; i < size; ++i)
        vec[i] = static_cast<double>(dist(rng));
}

void SharedData::putValuesIntoMatrix(
    int rows, int cols, std::vector<double>& mat, double min, double max, uint32_t seed
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(static_cast<int>(min), static_cast<int>(max));

    const int totalElements = rows * cols;
    for (int i = 0; i < totalElements; ++i)
        mat[i] = static_cast<double>(dist(rng));
}

void SharedData::putValuesIntoIdentityMatrix(int size, std::vector<double>& mat) {
    std::fill(mat.begin(), mat.end(), 0.0);
    for (int i = 0; i < size; ++i)
        mat[i * size + i] = 1.0;
}

void SharedData::preprocessA_T() {
    for (int i = 0; i < m; ++i) {
        if (constraintTypes[i] == ConstraintType::GREATEREQ) {
            for (int j = 0; j < n; ++j)
                A_T[j * m + i] = -A_T[j * m + i];
            A_T[(n + i) * m + i] = 1.0;
        } else {
            A_T[(n + i) * m + i] = +1.0;
        }
    }
}

void SharedData::preprocessB() {
    for (int i = 0; i < m; ++i) {
        if (constraintTypes[i] == ConstraintType::GREATEREQ)
            b[i] = -b[i];
    }
}

void SharedData::putValuesIntoA_T(
    std::vector<double>& A, double min, double max, uint32_t seed
) {
    putValuesIntoMatrix(n, m, A, min, max, seed);
    preprocessA_T();
}

void SharedData::putValuesIntoB(
    std::vector<double>& b, double min, double max, uint32_t seed
) {
    putValuesIntoVector(m, b, min, max, seed);
    preprocessB();
}

void SharedData::saveScalar(double& scalar, const std::vector<double>& vec, int i_pivot) {
    scalar = vec[i_pivot];
}

void SharedData::saveRow(std::vector<double>& vec_copy, int i_pivot) {
    const double* src = rowB_m1(i_pivot);
    std::copy(src, src + m, vec_copy.begin());
}

void SharedData::changeElement(std::vector<double>& vec, int i_pivot, int j_pivot) {
    basisIdx[i_pivot] = j_pivot;
    vec[i_pivot] = (j_pivot < n) ? c[j_pivot] : 0.0;
}
