#pragma once
#include <string>
#include <vector>
#include <iostream>

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
};
