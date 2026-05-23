#pragma once

#include <cstdint>
#include <vector>

enum class ConstraintType : uint8_t {
    LESSEREQ = 0,   // <=
    GREATEREQ = 1,  // >=
    EQUAL = 2,      // =
};

class SharedData {
    public:
        int n = 0;
        int m = 0;

        // Скаляри
        double d_ipRev = 0.0;     // 1/d_iPiv
        double x_B_iPiv = 0.0;    // змінна i_pivot вектора x_B

        // Вектори
        alignas(64) std::vector<double> b;      // розмір m x 1
        alignas(64) std::vector<double> c;      // розмір 1 x n
        alignas(64) std::vector<double> c_TB;   // розмір 1 x m
        alignas(64) std::vector<double> x_B;    // розмір m x 1

        alignas(64) std::vector<double> B_m1_iPiv;  // рядок i_pivot матриці B_m1 розмір 1 x m

        // Матриці
        alignas(64) std::vector<double> A;      // розмір m x n
        alignas(64) std::vector<double> B;      // розмір m x m
        alignas(64) std::vector<double> B_m1;   // розмір m x m

        std::vector<int> basisIdx; // індекси базисних змінних (розмір m)

        std::vector<ConstraintType> constraintTypes;

        SharedData() = default;
        ~SharedData() = default;

        bool needsDualStart = false;

        // Швидкий доступ до рядків матриць
        inline const double* rowA(int i) const { return A.data() + i * n; }
        inline       double* rowA(int i)       { return A.data() + i * n; }

        inline const double* rowB(int i) const { return B.data() + i * m; }
        inline       double* rowB(int i)       { return B.data() + i * m; }

        inline const double* rowB_m1(int i) const { return B_m1.data() + i * m; }
        inline       double* rowB_m1(int i)       { return B_m1.data() + i * m; }


        // Функції
        void initialise(int n, int m);
        void putValuesIntoVector(
            int size, std::vector<double> &vector, double min, double max, uint32_t seed
        );
        void putValuesIntoMatrix(
            int rowSize, int colSize, std::vector<double> &matrix, double min, double max, uint32_t seed
        );
        void putValuesIntoIdentityMatrix(
            int rowSize, int colSize, std::vector<double> &matrix
        );
        void preprocessA();
        void preprocessB();
        void putValuesIntoA(std::vector<double> &A, double min, double max, uint32_t seed);
        void putValuesIntoB(std::vector<double> &b, double min, double max, uint32_t seed);

        void saveScalar(double &scalar, std::vector<double> &vector, int i_pivot);
        void saveRow(std::vector<double> &vector_copy, int i_pivot);

        void changeElement(std::vector<double> &vector, int i_pivot, int j_pivot);
};
