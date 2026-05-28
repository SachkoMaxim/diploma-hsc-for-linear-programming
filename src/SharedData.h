#pragma once

#include <cstdint>
#include <vector>

enum class ConstraintType : uint8_t {
    LESSEREQ = 0,   // <=
    GREATEREQ = 1,  // >=
    EQUAL = 2,      // =
};

struct alignas(64) PaddedScalar {
    double value = 0.0;
    char   _pad[56]{}; // доповнення до 64 байт
};

class SharedData {
    public:
        int n = 0;
        int m = 0;

        // Скаляри
        PaddedScalar d_ipRev;       // 1/d[i_pivot]
        PaddedScalar x_B_iPiv;      // x_B[i_pivot] (копія перед оновленням)

        // Вектори
        alignas(64) std::vector<double> b;      // розмір m x 1
        alignas(64) std::vector<double> c;      // розмір 1 x n
        alignas(64) std::vector<double> c_TB;   // розмір 1 x m
        alignas(64) std::vector<double> x_B;    // розмір m x 1

        alignas(64) std::vector<double> B_m1_iPiv;  // рядок i_pivot матриці B_m1 розмір 1 x m

        // Матриці
        alignas(64) std::vector<double> A_T;    // розмір n x m (транспонована A розміром m x n)
        alignas(64) std::vector<double> B;      // розмір m x m
        alignas(64) std::vector<double> B_m1;   // розмір m x m

        // Службові
        std::vector<int> basisIdx; // індекси базисних змінних (розмір m)

        std::vector<ConstraintType> constraintTypes;
        bool needsDualStart = false;

        SharedData() = default;
        ~SharedData() = default;

        // Швидкий доступ до елементів матриць
        inline const double* colA_T(int j) const { return A_T.data() + j * m; }
        inline       double* colA_T(int j)       { return A_T.data() + j * m; }

        inline const double* rowB_m1(int i) const { return B_m1.data() + i * m; }
        inline       double* rowB_m1(int i)       { return B_m1.data() + i * m; }

        // Функції
        void initialise(int n_vars, int m_constraints);

        void putValuesIntoVector(
            int size, std::vector<double>& vec, double min, double max, uint32_t seed
        );
        void putValuesIntoMatrix(
            int rows, int cols, std::vector<double>& mat,
            double min, double max, uint32_t seed
        );
        void putValuesIntoIdentityMatrix(int size, std::vector<double>& mat);

        void putValuesIntoA_T(std::vector<double>& A, double min, double max, uint32_t seed);
        void putValuesIntoB(std::vector<double>& b, double min, double max, uint32_t seed);

        void saveScalar(double& scalar, const std::vector<double>& vec, int i_pivot);
        void saveRow(std::vector<double>& vec_copy, int i_pivot);
        void changeElement(std::vector<double>& vec, int i_pivot, int j_pivot);

    private:
        void preprocessA_T();
        void preprocessB();
};
