#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "ParallelSimplex.h"

class FileWriter {
    public:
        static inline void saveLargeResultToFile(
            const std::string& filename, SimplexStatus status, const std::vector<double>& solution,
            double resultZ, int iterations, int n_size, int m_size, int P,
            const std::vector<double>& b, const std::vector<double>& c, const std::vector<double>& A_T
        ) {
            std::ofstream outFile(filename);
            if (outFile.is_open()) {
                outFile << "Dimensions: n = " << n_size << ", m = " << m_size << ", P = " << P << "\n";
                outFile << "Iterations: " << iterations << "\n";;

                outFile << "Simplex Algorithm status: ";
                switch (status) {
                    case SimplexStatus::OPTIMAL:    outFile << "OPTIMAL\n";    break;
                    case SimplexStatus::UNBOUNDED:  outFile << "UNBOUNDED\n";  break;
                    case SimplexStatus::INFEASIBLE: outFile << "INFEASIBLE\n"; break;
                    case SimplexStatus::MAX_ITER:   outFile << "MAX_ITER\n";   break;
                    default:                        outFile << "UNKNOWN\n";    break;
                }

                outFile << "Z = " << resultZ << "\n";

                if (status == SimplexStatus::OPTIMAL) {
                    outFile << "Vector x: [ ";
                    for (double val : solution) outFile << val << " ";
                    outFile << "]\n";
                }

                outFile << "\n[INPUT DATA]\n";
                outFile << "Vector b: [ ";
                for (double val : b) outFile << val << " ";
                outFile << "]\n";

                outFile << "Vector c: [ ";
                for (double val : c) outFile << val << " ";
                outFile << "]\n";

                outFile << "Matrix A:\n";
                for (int i = 0; i < m_size; ++i) {
                    for (int j = 0; j < n_size; ++j) {
                        outFile << A_T[j * m_size + i] << "\t";
                    }
                    outFile << "\n";
                }

                outFile.close();
                std::cout << "[FILE LOG] All detailed data has been successfully saved to file '" << filename << "'\n";
            } else {
                std::cerr << "[ERROR] Could not open file '" << filename << "' for writing!\n";
            }
        }
};
