#include "SimplexMethod.h"

#include <limits>

#include "ConsoleVisualizer.h"
#include "SimplexMath.h"

SimplexMethod::SimplexMethod(SharedData &data_, int numThreads_, int maxIter_, uint32_t seed_)
    : data(data_), P(numThreads_), maxIter(maxIter_), seed(seed_)
{
    v.resize(data.m, 0.0);
    equationSolution.resize(data.n, 0.0);
}

SimplexStatus SimplexMethod::solve() {
    // Введення змінних
    data.putValuesIntoVector(data.m, data.b, 1.0, 50.0, seed);
    data.putValuesIntoVector(data.n, data.c, 1.0, 100.0, seed);
    std::fill(data.c_TB.begin(), data.c_TB.end(), 0.0);
    for (int i = 0; i < data.m; ++i) {
        data.b[i] += 1e-11;
    }
    data.x_B = data.b; // Оновлюємо початковий базис

    data.putValuesIntoA(data.A, -20.0, 20.0, seed);
    data.putValuesIntoIdentityMatrix(data.m, data.m, data.B);
    data.putValuesIntoIdentityMatrix(data.m, data.m, data.B_m1);

    ConsoleVisualizer::printVector("Vector b", data.b);
    ConsoleVisualizer::printVector("Vector c", data.c);
    ConsoleVisualizer::printVector("Vector c_TB", data.c_TB);
    ConsoleVisualizer::printVector("Vector x_B", data.x_B);
    ConsoleVisualizer::printMatrix("Matrix A", data.A, data.m, data.n);
    ConsoleVisualizer::printMatrix("Matrix B", data.B, data.m, data.m);
    ConsoleVisualizer::printMatrix("Identity Matrix B_m1", data.B_m1, data.m, data.m);
    std::cout << "Needs Dual Start: " << (data.needsDualStart ? "YES" : "NO") << "\n";

    iterations = 0;
    std::vector<double> c_j(data.n, 0.0);
    std::vector<double> d(data.m, 0.0);
    std::vector<double> t_j(data.m, 0.0);

    while (iterations < maxIter) {
        iterations++;

        // n = c_TB * B_m1
        SimplexMath::MultiplyVectorAndMatrix(data.c_TB, data.B_m1, v, data.m, data.m);

        // j_pivot = min(nA - c)
        j_pivot = -1;
        double min_delta = -1e-9;
        SimplexMath::MultiplyVectorAndMatrix(v, data.A, c_j, data.m, data.n);

        for (int j = 0; j < data.n; j++) {
            double delta = c_j[j] - data.c[j];

            if (delta < min_delta) {
                min_delta = delta;
                j_pivot = j;
            }
        }

        if (j_pivot == -1) {
            std::cout << "Optimal solution found in " << iterations << " iterations.\n";
            break;
        }

        // d = B_m1 * A(j_pivot)
        SimplexMath::MultiplyMatrixAndColumn(data.B_m1, data.A, j_pivot, d, data.m, data.n);

        // i_pivot = min(x_B/d) ЗА ПРАВИЛОМ ГАРРІСА
        i_pivot = -1;
        double harris_epsilon = 1e-7;
        double min_pivot_step = std::numeric_limits<double>::max();

        // ПРОХІД 1: Визначення максимально допустимиго кроку (max allowable pivot step)
        for (int i = 0; i < data.m; i++) {
            if (d[i] > 1e-9) {
                // Дозволяємо x_B бути злегка від'ємним у межах похибки
                double max_step = (data.x_B[i] + harris_epsilon) / d[i];
                if (max_step < min_pivot_step) {
                    min_pivot_step = max_step;
                }
            }
        }

        // Якщо min_pivot_step не змінився, задача необмежена
        if (min_pivot_step == std::numeric_limits<double>::max()) {
            std::cout << "Target function is unbounded.\n";
            return SimplexStatus::UNBOUNDED;
        }

        // ПРОХІД 2: Серед усіх рядків, що задовольняють min_pivot_step,
        // вибір того, у якого НАЙБІЛЬШИЙ знаменник d[i] (для чисельної стабільності)
        double max_d_val = -1.0;

        for (int i = 0; i < data.m; i++) {
            if (d[i] > 1e-9) {
                double actual_step = data.x_B[i] / d[i];

                // Перевірка, чи входить рядок у розширену зону допустимості
                if (actual_step <= min_pivot_step) {
                    // Вибір рядка з найбільшим d[i], щоб уникнути ділення на малі числа
                    if (d[i] > max_d_val - 1e-9) {
                        max_d_val = d[i];
                        i_pivot = i;
                    }
                }
            }
        }

        // c_TB(j_pivot) = c(j_pivot)
        data.changeElement(data.c_TB, i_pivot, j_pivot);

        // d_ipRev = 1 / d(i_pivot)
        data.d_ipRev = 1 / d[i_pivot];

        // x_B_iPiv = x_B(i_pivot)
        data.saveScalar(data.x_B_iPiv, data.x_B, i_pivot);

        // B_m1_iPiv = B_m1(i_pivot)
        data.saveRow(data.B_m1_iPiv, i_pivot);

        // Оновлення B_m1
        SimplexMath::UpdateMatrix(data.B_m1_iPiv, d, data.d_ipRev, data.B_m1, data.m, data.m, i_pivot);

        // Оновлення x_B
        SimplexMath::UpdateVector(d, data.x_B_iPiv, data.d_ipRev, data.x_B, data.m, i_pivot);

        ConsoleVisualizer::printVector("Vector x_B", data.x_B);
        ConsoleVisualizer::printVector("Vector c_TB", data.c_TB);
        ConsoleVisualizer::printMatrix("Identity Matrix B_m1", data.B_m1, data.m, data.m);
    }

    // Якщо вийшли з циклу по break (знайшли оптимальний розв'язок)
    if (iterations < maxIter) {
        std::fill(equationSolution.begin(), equationSolution.end(), 0.0);
        for (int i = 0; i < data.m; ++i) {
            int idx = data.basisIdx[i];
            if (idx < data.n) equationSolution[idx] = data.x_B[i];
        }

        result = 0.0;
        for (int j = 0; j < data.n; ++j) result += data.c[j] * equationSolution[j];

        return SimplexStatus::OPTIMAL;
    }

    std::cout << "Iteration limit reached.\n";
    return SimplexStatus::MAX_ITER;
}
