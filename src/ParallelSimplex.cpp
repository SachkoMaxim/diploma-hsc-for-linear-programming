#include "ParallelSimplex.h"

#include <limits>
#include <omp.h>
#include <oneapi/tbb.h>

#include "ConsoleVisualizer.h"
#include "SimplexMath.h"

ParallelSimplex::ParallelSimplex(SharedData &data_, int numThreads_, int maxIter_, uint32_t seed_)
    : data(data_), P(numThreads_), maxIter(maxIter_), arena(numThreads_), seed(seed_)
{
    v.resize(data.m, 0.0);
    equationSolution.resize(data.n, 0.0);
}

SimplexStatus ParallelSimplex::solve() {
    int T_id;


#pragma omp parallel num_threads(P) private(T_id) shared(data, seed)
    {
        // Отримання номеру потоку
        T_id = omp_get_thread_num();

        // Виведення інформації про початок роботи потоку
#pragma omp critical
        {
            std::cout << "Task T" << T_id + 1 << " is started\n";
        }

        // Введення даних
        switch (T_id)
        {
            case 0: // T1: Введення A
                data.putValuesIntoA_T(data.A_T, -20.0, 20.0, seed);
                break;
            case 1: // T2: Введення b, c, c_TB, x_B
                data.putValuesIntoB(data.b, 1.0, 50.0, seed);
                data.putValuesIntoVector(data.n, data.c, 1.0, 100.0, seed);
                std::fill(data.c_TB.begin(), data.c_TB.end(), 0.0);

                for (int i = 0; i < data.m; ++i) data.b[i] += 1e-11;
                break;
            case 2: // T3: Введення B
                data.putValuesIntoIdentityMatrix(data.m, data.m, data.B);
                break;
            case 3: // T4: Введення B_m1
                data.putValuesIntoIdentityMatrix(data.m, data.m, data.B_m1);
                break;
            default:
                break;
        }

        // Бар'єр B1 для синхронізації по введенню даних
#pragma omp barrier

#pragma omp single
        {
            data.x_B = data.b; // Оновлюємо початковий базис
            // Скидаємо прапорець і аналізуємо знаки спільно з оновленням типів обмежень
            data.needsDualStart = false;
            for (int i = 0; i < data.m; ++i) {
                if (data.b[i] < 0.0) data.needsDualStart = true;
                data.constraintTypes[i] = ConstraintType::LESSEREQ;
            }
        }
    }

    ConsoleVisualizer::printVector("Vector b", data.b);
    ConsoleVisualizer::printVector("Vector c", data.c);
    ConsoleVisualizer::printVector("Vector c_TB", data.c_TB);
    ConsoleVisualizer::printVector("Vector x_B", data.x_B);
    ConsoleVisualizer::printMatrix("Matrix A_T", data.A_T, data.n, data.m);
    ConsoleVisualizer::printMatrix("Matrix B", data.B, data.m, data.m);
    ConsoleVisualizer::printMatrix("Identity Matrix B_m1", data.B_m1, data.m, data.m);
    std::cout << "Needs Dual Start: " << (data.needsDualStart ? "YES" : "NO") << "\n";

    iterations = 0;
    std::vector<double> d(data.m, 0.0);
    std::vector<double> u(data.n, 0.0);

    // Дуальна частина (Прибирання від'ємних b / x_B)
    if (data.needsDualStart) {
        // Зберігання оригінального вектора цілей і створення штучного нульового базису
        std::vector<double> original_c = data.c;
        std::fill(data.c.begin(), data.c.end(), 0.0); // Тимчасово анулюються ціни

        while (iterations < maxIter) {
            ++iterations;

            // Вибір i_pivot за рядком з мінімальним від'ємним x_B[i]
            i_pivot = -1;
            double min_x_B = -1e-9;
            for (int i = 0; i < data.m; ++i) {
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
            SimplexMath::MultiplyVectorAndTransposedMatrix(data.B_m1_iPiv, data.A_T, u, data.m, data.n);

            // Обчислення1-4 відносно ШТУЧНОЇ (нульової) цільової функції;
            arena.execute([&]() {
                computeVectorV();
                computeJPivotForDual(u);
            });

            if (j_pivot == -1) {
                std::cout << "The problem is INFEASIBLE (No dual pivot column found).\n";
                return SimplexStatus::INFEASIBLE;
            }

            // Обчислення5
            arena.execute([&]() {
                // Обчислення5
                computeVectorD(d);
            });

            // Обчислення8-12
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
        ++iterations;

        // Обчислення1-4
        arena.execute([&]() {
            computeVectorV();
            computeJPivotForPrimal();
        });

        if (j_pivot == -1) {
            std::cout << "Optimal solution found in " << iterations << " iterations.\n";
            break;
        }

        // Обчислення5 d = B_m1 * A(j_pivot)н
        arena.execute([&]() {
            computeVectorD(d);
        });

        // Обчислення6-7 ЗА ПРАВИЛОМ ГАРРІСА
        i_pivot = harrisRatioPivot(d);

        if (i_pivot == -1) {
            std::cout << "Target function is unbounded.\n";
            return SimplexStatus::UNBOUNDED;
        }

        // Обчислення8-12
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

void ParallelSimplex::computeVectorV() {
    v = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.m), // Діапазон рядків H
        std::vector<double>(data.m, 0.0),   // Ініціалізація локального vi
        // Обчислення1: v = c_TBн * B_m1н
        [&](const tbb::blocked_range<int> &r, std::vector<double> local_v) {
            for (int i = r.begin(); i != r.end(); ++i) {
                double c_val = data.c_TB[i];
                if (std::abs(c_val) < 1e-12) continue;

                for (int j = 0; j < data.m; ++j) {
                    local_v[j] += c_val * data.B_m1[i * data.m + j];
                }
            }
            return local_v;
        },
        // Обчислення2: v = v + vi (КД1)
        [](std::vector<double> main_v, const std::vector<double> &local_v) {
            for (size_t i = 0; i < main_v.size(); ++i) main_v[i] += local_v[i];
            return main_v;
        }
    );
}

void ParallelSimplex::computeJPivotForPrimal() {
    j_pivot = -1;
    std::pair<double, int> identity_j = { -1e-9, -1 };

    j_pivot = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.n),
        identity_j,
        // Обчислення3: ji = min(v * Aн - cн)
        [&](const tbb::blocked_range<int>& r, std::pair<double, int> local) {
            for (int j = r.begin(); j < r.end(); ++j) {
                double v_Aj = 0.0;
                const int col_offset = j * data.m;

                for (int i = 0; i < data.m; ++i) {
                    v_Aj += v[i] * data.A_T[col_offset + i];
                }

                double delta = v_Aj - data.c[j];

                if (delta < local.first) {
                    local.first = delta;
                    local.second = j;
                }
                else if (std::abs(delta - local.first) < 1e-11 && local.second != -1 && j < local.second) {
                    local.second = j;
                }
            }
            return local;
        },
        // Обчислення4: j_pivot = min(j_pivot, ji) (КД2)
        [](std::pair<double, int> a, std::pair<double, int> b) {
            if (a.second == -1) return b;
            if (b.second == -1) return a;

            if (a.first < b.first) return a;
            if (b.first < a.first) return b;

            // Якщо дельти рівні, то вибір ведучого j_pivot за правилом Бленда / абсолютного мінімуму
            return (a.second < b.second) ? a : b;
        }
    ).second;
}

void ParallelSimplex::computeJPivotForDual(const std::vector<double> &u) {
    j_pivot = -1;
    std::pair<double, int> identity_dual = { std::numeric_limits<double>::max(), -1 };

    this->j_pivot = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.n),
        identity_dual,
        // Лямбда 1: Обчислення локального мінімуму u в піддіапазоні стовпців
        [&](const tbb::blocked_range<int>& r, std::pair<double, int> local) {
            for (int j = r.begin(); j < r.end(); ++j) {
                if (u[j] < -1e-9) {
                    double v_Aj = 0.0;
                    const int col_offset = j * data.m;

                    for (int i = 0; i < data.m; ++i) {
                        v_Aj += v[i] * data.A_T[col_offset + i];
                    };

                    double abs_theta = std::abs((v_Aj - data.c[j]) / u[j]);

                    if (abs_theta < local.first) {
                        local.first = abs_theta;
                        local.second = j;
                    }
                    else if (std::abs(abs_theta - local.first) < 1e-11 && local.second != -1 && j < local.second) {
                        local.second = j;
                    }
                }
            }
            return local;
        },
        [](std::pair<double, int> a, std::pair<double, int> b) {
            if (a.second == -1) return b;
            if (b.second == -1) return a;

            if (a.first < b.first) return a;
            if (b.first < a.first) return b;

            // Якщо дельти рівні, то вибір ведучого j_pivot за правилом Бленда / абсолютного мінімуму
            return (a.second < b.second) ? a : b;
        }
    ).second;
}

void ParallelSimplex::computeVectorD(std::vector<double> &d) {
    //Обчислення5 d = B_m1 * A(j_pivot)н
    tbb::parallel_for(tbb::blocked_range<int>(0, data.m), [&](const tbb::blocked_range<int> &r) {
        const double* col = data.A_T.data() + j_pivot * data.m;
        for (int i = r.begin(); i < r.end(); ++i) {
            double sum = 0.0;
            const double* row = data.B_m1.data() + i * data.m;
            for (int j = 0; j < data.m; ++j) {
                sum += row[j] * col[j];
            }
            d[i] = sum;
        }
    });
}

void ParallelSimplex::performBasisUpdate(const std::vector<double> &d,
                                       int ip, int jp)
{
    arena.execute([&]() {
        tbb::parallel_invoke(
            // Обчислення8 c_TB(j_pivot) = c(j_pivot)
            [&]() { data.changeElement(data.c_TB, ip, jp); },

            // Обчислення9 d_ipRev = 1 / d(i_pivot)
            [&]() { data.d_ipRev = 1.0 / d[ip]; },

            // Копіювання x_B_iPiv = x_B(i_pivot) (КД4)
            [&]() { data.saveScalar(data.x_B_iPiv, data.x_B, ip); },

            // Копіювання B_m1_iPiv = B_m1(i_pivot) (КД5)
            [&]() { data.saveRow(data.B_m1_iPiv, ip); });
        });

    // Обчислення fн та Оновлення B_m1 з x_B
    SimplexMath::UpdateMatrixAndVectorParallel(
        data.B_m1_iPiv, d, data.d_ipRev, data.B_m1, data.x_B_iPiv, data.x_B, data.m, data.m, ip
    );
}

int ParallelSimplex::harrisRatioPivot(const std::vector<double> &d) const
{
    constexpr double harris_epsilon = 1e-7;
    double global_min_step = std::numeric_limits<double>::max();

    arena.execute([&]() {
        // ПРОХІД 1: Визначення максимально допустимиго кроку (max allowable pivot step)
        global_min_step = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, data.m),
            std::numeric_limits<double>::max(), // Стартове значення для пошуку мінімуму

            // Обчислення6 ii = min((x_B)н / dн)
            [&](const tbb::blocked_range<int>& r, double local_min_step) -> double {
                for (int i = r.begin(); i < r.end(); ++i) {
                    if (d[i] > 1e-9) {
                        // Дозволяємо x_B бути злегка від'ємним у межах похибки
                        local_min_step = std::min(local_min_step, (data.x_B[i] + harris_epsilon) / d[i]);
                    }
                }
                return local_min_step;
            },
            // Обчислення7 i_pivot = min(i_pivot, ii) (КД3)
            [](double a, double b) {
                return std::min(a, b);
            }
        );
    });

    // Перевірка на необмеженість задачі лінійного програмування
    if (global_min_step == std::numeric_limits<double>::max()) return -1;

    int final_ip = -1;

    arena.execute([&]() {
        // ПРОХІД 2: Вибір рядка з найбільшим знаменником d[i]
        std::pair<double, int> identity_harris_2 = { -1.0, -1 }; // {max_d, ip}

        auto best_row = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, data.m),
            identity_harris_2,
            // Обчислення6 ii = min((x_B)н / dн)
            [&](const tbb::blocked_range<int>& r, std::pair<double, int> local) {
                for (int i = r.begin(); i < r.end(); ++i) {
                    // Умова чисельної стабільності Гарріса
                    if (d[i] > 1e-9 && data.x_B[i] / d[i] <= global_min_step && d[i] > local.first - 1e-9) {
                        local.first = d[i];
                        local.second = i;
                    }
                }
                return local;
            },
            // Обчислення7 i_pivot = min(i_pivot, ii) (КД3)
            [](std::pair<double, int> a, std::pair<double, int> b) {
                if (a.second == -1) return b;
                if (b.second == -1) return a;
                // Обираємо потік, який знайшов НАЙБІЛЬШИЙ знаменник d[i]
                return (a.first > b.first) ? a : b;
            }
        );

        final_ip = best_row.second;
    });

    return final_ip;
}
