#pragma once
#include <string>
#include <vector>
#include <iostream>

#include "ParallelSimplex.h"

class ConsoleVisualizer {
    public:
        // Універсальний друк будь-якого вектора double
        static inline void printVector(const std::string& name, const std::vector<double>& vec) {
            std::cout << name << ": [ ";
            for (double val : vec) std::cout << val << " ";
            std::cout << "]\n";
        }

        // Універсальний друк будь-якої одновимірної матриці (flatted matrix)
        static inline void printMatrix(const std::string& name, const std::vector<double>& mat, int rows, int cols) {
            std::cout << name << ":\n";
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    std::cout << mat[i * cols + j] << "\t";
                }
                std::cout << "\n";
            }
        }

        // Виведення транспонованої матриці "на льоту" (без нової змінної)
        static inline void printMatrixTransposed(
            const std::string& name, const std::vector<double>& mat, int rows, int cols
        ) {
            std::cout << name << ":\n";
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < rows; ++j) {
                    std::cout << mat[j * cols + i] << "\t";
                }
                std::cout << "\n";
            }
        }

        static inline void printSimplexStatus(const std::string& name, SimplexStatus s) {
            std::cout << name << ": ";
            switch (s) {
                case SimplexStatus::OPTIMAL:    std::cout << "OPTIMAL\n";    break;
                case SimplexStatus::UNBOUNDED:  std::cout << "UNBOUNDED\n";  break;
                case SimplexStatus::INFEASIBLE: std::cout << "INFEASIBLE\n"; break;
                case SimplexStatus::MAX_ITER:   std::cout << "MAX_ITER\n";   break;
                default:                        std::cout << "UNKNOWN\n";    break;
            }
        }
};
