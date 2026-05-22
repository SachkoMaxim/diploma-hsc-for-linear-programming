#include <iostream>

#include "src/SharedData.h"

const int n = 4;
const int m = 4;
const int P = 4;

const uint32_t SEED = 77;

using namespace std;

int main() {

    SharedData data;
    data.initialise(n, m);

    data.putValuesIntoVector(m, data.b, 1.0, 100.0, SEED);
    data.putValuesIntoVector(n, data.c, 1.0, 50.0, SEED);
    data.putValuesIntoVector(m, data.c_TB, 1.0, 100.0, SEED);
    data.putValuesIntoVector(m, data.x_B, 1.0, 100.0, SEED);

    data.putValuesIntoA(data.A, -20.0, 20.0, SEED);
    data.putValuesIntoIdentityMatrix(m, m, data.B);
    data.putValuesIntoIdentityMatrix(m, m, data.B_m1);

    auto printVector = [](const string& name, const vector<double>& vec) {
        cout << name << ": [ ";
        for (double val : vec) cout << val << " ";
        cout << "]\n";
    };

    auto printMatrix = [](const string& name, const vector<double>& mat, int rows, int cols) {
        cout << name << ":\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << mat[i * cols + j] << "\t";
            }
            cout << "\n";
        }
    };

    printVector("Vector b", data.b);
    printVector("Vector c", data.c);
    printVector("Vector c_TB", data.c_TB);
    printVector("Vector x_B", data.x_B);
    printMatrix("Matrix A", data.A, m, n);
    printMatrix("Matrix B", data.B, m, m);
    printMatrix("Identity Matrix B_m1", data.B_m1, m, m);
    cout << "Needs Dual Start: " << (data.needsDualStart ? "YES" : "NO") << "\n";

    const auto lang = "C++";
    std::cout << "Hello and welcome to " << lang << "!\n";

    return 0;
}
