#include "../src/SimplexMethod.h"

#include <limits>

#include "../src/ConsoleVisualizer.h"
#include "../src/SimplexMath.h"

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
    for (int i = 0; i < data.m; ++i) data.b[i] += 1e-11;

    data.putValuesIntoA(data.A, -20.0, 20.0, seed);
    data.x_B = data.b; // Оновлюємо початковий базис
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
    std::vector<double> u(data.n, 0.0);

    // Дуальна частина (Прибирання від'ємних b / x_B)
    if (data.needsDualStart) {
        // Зберігання оригінального вектора цілей і створення штучного нульового базису
        std::vector<double> original_c = data.c;
        std::fill(data.c.begin(), data.c.end(), 0.0); // Тимчасово анулюються ціни

        while (iterations < maxIter) {
            iterations++;

            // Вибір i_pivot за рядком з мінімальним від'ємним x_B[i]
            i_pivot = -1;
            double min_x_B = -1e-9;
            for (int i = 0; i < data.m; i++) {
                if (data.x_B[i] < min_x_B) {
                    min_x_B = data.x_B[i];
                    i_pivot = i;
                }
            }

            if (i_pivot == -1) {
                std::cout << "Dual Phase: Feasible basis found in " << iterations << " iterations.\n";
                break;
            }

            // u = рядок i_pivot матриці B_m1 * A
            data.saveRow(data.B_m1_iPiv, i_pivot);
            SimplexMath::MultiplyVectorAndMatrix(data.B_m1_iPiv, data.A, u, data.m, data.n);

            // Обчислення дельти відносно ШТУЧНОЇ (нульової) цільової функції
            computeReducedCosts(c_j);

            std::cout << "Dual iter " << iterations << ", i_pivot=" << i_pivot << "\n";
            ConsoleVisualizer::printVector("u", u);
            ConsoleVisualizer::printVector("c_j", c_j);
            ConsoleVisualizer::printVector("x_B", data.x_B);

            // Вибір ведучого j_pivot за правилом Бленда / абсолютного мінімуму
            j_pivot = -1;
            double min_abs_theta = std::numeric_limits<double>::max();
            for (int j = 0; j < data.n; j++) {
                if (u[j] < -1e-9) {
                    double abs_theta = std::abs((c_j[j] - data.c[j]) / u[j]);
                    if (abs_theta < min_abs_theta) {
                        min_abs_theta = abs_theta;
                        j_pivot = j;
                    }
                }
            }

            if (j_pivot == -1) {
                std::cout << "The problem is INFEASIBLE (No dual pivot column found).\n";
                return SimplexStatus::INFEASIBLE;
            }

            // Обчислення d
            SimplexMath::MultiplyMatrixAndColumn(data.B_m1, data.A, j_pivot, d, data.m, data.n);

            performBasisUpdate(d, i_pivot, j_pivot);
        }

        // Відновлення оригінальної цільової функції
        data.c = original_c;

        for (int i = 0; i < data.m; ++i) {
            int idx = data.basisIdx[i];
            data.c_TB[i] = (idx < data.n) ? data.c[idx] : 0.0;
        }
    }

    // Пряма частина (Виведення задачі до Максимуму)
    while (iterations < maxIter) {
        iterations++;

        // n = c_TB * B_m1
        computeReducedCosts(c_j);

        // j_pivot = min(nA - c)
        j_pivot = -1;
        double min_delta = -1e-9;

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
        i_pivot = harrisRatioPivot(d);

        if (i_pivot == -1) {
            std::cout << "Target function is unbounded.\n";
            return SimplexStatus::UNBOUNDED;
        }

        performBasisUpdate(d, i_pivot, j_pivot);

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

void SimplexMethod::computeReducedCosts(std::vector<double> &c_j) {
    // n = c_TB * B_m1
    SimplexMath::MultiplyVectorAndMatrix(data.c_TB, data.B_m1, v, data.m, data.m);

    // c_j = n * A
    SimplexMath::MultiplyVectorAndMatrix(v, data.A, c_j, data.m, data.n);
}

void SimplexMethod::performBasisUpdate(const std::vector<double> &d,
                                       int ip, int jp)
{
    // c_TB(j_pivot) = c(j_pivot)
    data.changeElement(data.c_TB, ip, jp);

    // d_ipRev = 1 / d(i_pivot)
    data.d_ipRev = 1.0 / d[ip];

    // x_B_iPiv = x_B(i_pivot)
    data.saveScalar(data.x_B_iPiv, data.x_B, ip);

    // B_m1_iPiv = B_m1(i_pivot)
    data.saveRow(data.B_m1_iPiv, ip);

    // Оновлення B_m1
    SimplexMath::UpdateMatrix(data.B_m1_iPiv, d, data.d_ipRev, data.B_m1, data.m, data.m, ip);

    // Оновлення x_B
    SimplexMath::UpdateVector(d, data.x_B_iPiv, data.d_ipRev, data.x_B, data.m, ip);
}

int SimplexMethod::harrisRatioPivot(const std::vector<double> &d) const
{
    constexpr double harris_epsilon = 1e-7;
    double min_step = std::numeric_limits<double>::max();

    // ПРОХІД 1: Визначення максимально допустимиго кроку (max allowable pivot step)
    for (int i = 0; i < data.m; i++)
        if (d[i] > 1e-9)
            // Дозволяємо x_B бути злегка від'ємним у межах похибки
            min_step = std::min(min_step, (data.x_B[i] + harris_epsilon) / d[i]);

    if (min_step == std::numeric_limits<double>::max())
        return -1; // задача необмежена

    // ПРОХІД 2: Серед усіх рядків, що задовольняють min_pivot_step,
    // вибір того, у якого НАЙБІЛЬШИЙ знаменник d[i] (для чисельної стабільності)
    int ip = -1;
    double max_d = -1.0;

    for (int i = 0; i < data.m; i++) {
        if (d[i] > 1e-9 && data.x_B[i] / d[i] <= min_step && d[i] > max_d - 1e-9) {
            max_d = d[i];
            ip = i;
        }
    }
    return ip;
}
