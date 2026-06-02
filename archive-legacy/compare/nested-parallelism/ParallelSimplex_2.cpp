#include "ParallelSimplex.h"

#include <limits>
#include <iostream>
#include <algorithm>
#include <oneapi/tbb.h>
#include <omp.h>

#include "SimplexMath.h"

ParallelSimplex::ParallelSimplex(SharedData& data_, int numThreads_, int maxIter_, uint32_t seed_)
    : data(data_), P(numThreads_), maxIter(maxIter_), arena(numThreads_), seed(seed_)
{
    v.resize(data.m, 0.0);
    equationSolution.resize(data.n, 0.0);
}

SimplexStatus ParallelSimplex::solve() {
    return arena.execute([&]() -> SimplexStatus {
        tbb::parallel_invoke(
            // T1: Введення A
            [&]() { data.putValuesIntoA_T(data.A_T, -20.0, 20.0, seed); },
            // T2: Введення b, c, c_TB + x_B = b
            [&]() {
                data.putValuesIntoB(data.b, 1.0, 50.0, seed);
                data.putValuesIntoVector(data.n, data.c, 1.0, 100.0, seed);
                std::fill(data.c_TB.begin(), data.c_TB.end(), 0.0);
                for (int i = 0; i < data.m; ++i) data.b[i] += 1e-11;

                data.x_B = data.b;
            },
            // T3: Введення B
            [&]() { data.putValuesIntoIdentityMatrix(data.m, data.B); },
            // T4: Введення B_m1
            [&]() { data.putValuesIntoIdentityMatrix(data.m, data.B_m1); }
        );

        data.needsDualStart = false;
        for (int i = 0; i < data.m; ++i) {
            if (data.b[i] < 0.0) { data.needsDualStart = true; break; }
        }
        std::cout << "Needs Dual Start: " << (data.needsDualStart ? "YES" : "NO") << "\n";

        iterations = 0;
        std::vector<double> d(data.m, 0.0);
        std::vector<double> u(data.n + data.m, 0.0);

        // Дуальна фаза: прибирання від'ємних x_B[i]
        if (data.needsDualStart) {
            std::vector<double> original_c = data.c;
            std::fill_n(data.c.begin(), data.n, -1.0); // тимчасово цільова функція з -1

            while (iterations < maxIter) {
                ++iterations;

                i_pivot = computeIPivotForDual();
                if (i_pivot == -1) {
                    std::cout << "Dual Phase: Feasible basis found in "
                              << iterations << " iterations.\n";
                    break;
                }

                // u = B_m1[i_pivot] * A
                data.saveRow(data.B_m1_iPiv, i_pivot);
                SimplexMath::MultiplyRowAndTransposedMatrix(
                    data.B_m1_iPiv, data.A_T, u, data.m, data.n + data.m
                );

                // Обчислення1-4: v та j_pivot для двоїстого критерію
                computeVectorV();
                j_pivot = computeJPivot(
                    { std::numeric_limits<double>::max(), -1 },
                    [&](int j, double v_Aj) -> std::pair<bool, double> {
                        if (u[j] >= -1e-6) return { false, 0.0 };
                        double c_j = (j < data.n) ? data.c[j] : 0.0;
                        return { true, std::abs((v_Aj - c_j) / u[j]) };
                    }
                );

                if (j_pivot == -1) {
                    std::cout << "Problem is INFEASIBLE (no dual pivot column).\n";
                    return SimplexStatus::INFEASIBLE;
                }

                // Обчислення5: d = B^-1 * A[j_pivot]
                computeVectorD(d);

                // Обчислення8-12: оновлення базису
                performBasisUpdate(d, i_pivot, j_pivot);
            }

            data.c = original_c;
            // Відновлення c_TB після повернення оригінальної цільової функції
            restoreCTB();
        }

        // Пряма фаза: рух до максимуму цільової функції
        while (iterations < maxIter) {
            ++iterations;

            // Обчислення1-4: v та j_pivot для прямого критерію
            computeVectorV();
            j_pivot = computeJPivot(
                { -1e-6, -1 },
                [&](int j, double v_Aj) -> std::pair<bool, double> {
                    double c_j = (j < data.n) ? data.c[j] : 0.0;
                    const double delta = v_Aj - c_j;
                    return { delta < -1e-6, delta };
                }
            );

            if (j_pivot == -1) {
                std::cout << "Optimal solution found in " << iterations << " iterations.\n";
                break;
            }

            // Обчислення5 d = B_m1 * A(j_pivot)н
            computeVectorD(d);

            // Обчислення6-7: вибір i_pivot за правилом Гарріса
            i_pivot = harrisRatioPivot(d);
            if (i_pivot == -1) {
                std::cout << "Objective function is unbounded.\n";
                return SimplexStatus::UNBOUNDED;
            }

            // Обчислення8-12: оновлення базису
            performBasisUpdate(d, i_pivot, j_pivot);
        }

        if (iterations >= maxIter) {
            std::cout << "Iteration limit reached.\n";
            return SimplexStatus::MAX_ITER;
        }

        // Збір розв'язку
        std::fill(equationSolution.begin(), equationSolution.end(), 0.0);
        for (int i = 0; i < data.m; ++i) {
            const int idx = data.basisIdx[i];
            if (idx < data.n) equationSolution[idx] = data.x_B[i];
        }

        result = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, data.m),
            0.0,
            [&](const tbb::blocked_range<int>& inner_r, double inner_sum) {
                for (int i = inner_r.begin(); i < inner_r.end(); ++i)
                    inner_sum += v[i] * data.b[i];
                return inner_sum;
            },
            std::plus<double>()
        );

        return SimplexStatus::OPTIMAL;
    }); // кінець arena.execute - потоки вивільняються тут
}

void ParallelSimplex::computeVectorV() {
    v = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.m),
        std::vector<double>(data.m, 0.0),
        [&](const tbb::blocked_range<int>& r, std::vector<double> local_v) {
            for (int i = r.begin(); i != r.end(); ++i) {
                const double c_val = data.c_TB[i];
                if (std::abs(c_val) < 1e-12) continue;
                const double* row = data.B_m1.data() + i * data.m;

                // Вкладений паралельний цикл для додавання елементів
                tbb::parallel_for(tbb::blocked_range<int>(0, data.m), [&](const tbb::blocked_range<int>& inner_r) {
                    for (int j = inner_r.begin(); j < inner_r.end(); ++j) {
                        local_v[j] += c_val * row[j];
                    }
                });
            }
            return local_v;
        },
        [](std::vector<double> a, const std::vector<double>& b) {
            // Вкладений parallel_for навіть для редукції (злиття) векторів локальних результатів
            tbb::parallel_for(tbb::blocked_range<int>(0, a.size()), [&](const tbb::blocked_range<int>& inner_r) {
                for (size_t i = inner_r.begin(); i < inner_r.end(); ++i) a[i] += b[i];
            });
            return a;
        }
    );
}

template<typename InnerFn>
int ParallelSimplex::computeJPivot(std::pair<double, int> identity, InnerFn&& innerFn) {
    auto join = [](std::pair<double, int> a, std::pair<double, int> b) {
        if (a.second == -1) return b;
        if (b.second == -1) return a;
        if (a.first < b.first) return a;
        if (b.first < a.first) return b;
        return (a.second < b.second) ? a : b;
    };

    return tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.n + data.m),
        identity,
        [&](const tbb::blocked_range<int>& r, std::pair<double, int> local) {
            for (int j = r.begin(); j < r.end(); ++j) {
                double v_Aj = 0.0;

                if (j < data.n) {
                    const double* col = data.A_T.data() + j * data.m;
                    // Вкладений parallel_reduce замість SIMD скалярного добутку
                    v_Aj = tbb::parallel_reduce(
                        tbb::blocked_range<int>(0, data.m),
                        0.0,
                        [&](const tbb::blocked_range<int>& inner_r, double inner_sum) {
                            for (int i = inner_r.begin(); i < inner_r.end(); ++i)
                                inner_sum += v[i] * col[i];
                            return inner_sum;
                        },
                        std::plus<double>()
                    );
                } else {
                    v_Aj = v[j - data.n];
                }

                auto [is_candidate, value] = innerFn(j, v_Aj);
                if (!is_candidate) continue;

                if (local.second == -1 || value < local.first) {
                    local = { value, j };
                } else if (std::abs(value - local.first) < 1e-11) {
                    if (j < local.second) local.second = j;
                }
            }
            return local;
        },
        join
    ).second;
}

void ParallelSimplex::computeVectorD(std::vector<double>& d) {
    tbb::parallel_for(tbb::blocked_range<int>(0, data.m),
        [&](const tbb::blocked_range<int>& r) {
            if (j_pivot < data.n) {
                const double* col = data.A_T.data() + j_pivot * data.m;
                for (int i = r.begin(); i < r.end(); ++i) {
                    const double* row = data.B_m1.data() + i * data.m;
                    // Вкладена редукція для обчислення d[i]
                    d[i] = tbb::parallel_reduce(
                        tbb::blocked_range<int>(0, data.m),
                        0.0,
                        [&](const tbb::blocked_range<int>& inner_r, double inner_sum) {
                            for (int j = inner_r.begin(); j < inner_r.end(); ++j)
                                inner_sum += row[j] * col[j];
                            return inner_sum;
                        },
                        std::plus<double>()
                    );
                }
            } else {
                int target_col = j_pivot - data.n;
                for (int i = r.begin(); i < r.end(); ++i) {
                    d[i] = data.B_m1[i * data.m + target_col];
                }
            }
        }
    );
}

void ParallelSimplex::performBasisUpdate(const std::vector<double>& d, int ip, int jp) {
    tbb::parallel_invoke(
        // Обчислення8 c_TB(j_pivot) = c(j_pivot)
        [&]() { data.changeElement(data.c_TB, ip, jp); },
        // Обчислення9 d_ipRev = 1 / d(i_pivot)
        [&]() { data.d_ipRev.value = 1.0 / d[ip]; },
        // Копіювання x_B_iPiv = x_B(i_pivot) (КД4)
        [&]() { data.saveScalar(data.x_B_iPiv.value, data.x_B, ip); },
        // Копіювання B_m1_iPiv = B_m1(i_pivot) (КД5)
        [&]() { data.saveRow(data.B_m1_iPiv, ip); }
    );

    // Обчислення10-12: оновлення B^-1 та x_B "на місці"
    SimplexMath::UpdateMatrixAndVectorParallel(
        data.B_m1_iPiv, d, data.d_ipRev.value,
        data.B_m1, data.x_B_iPiv.value, data.x_B,
        data.m, data.m, ip
    );
}

int ParallelSimplex::harrisRatioPivot(const std::vector<double>& d) const {
    constexpr double harris_eps = 1e-7;
    constexpr double pivot_threshold = 1e-6;

    // Прохід 1: мінімально допустимий крок theta_min
    const double theta_min = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.m),
        std::numeric_limits<double>::max(),
        // Обчислення6 ii = min((x_B)н / dн)
        [&](const tbb::blocked_range<int>& r, double local_min) {
            for (int i = r.begin(); i < r.end(); ++i) {
                if (d[i] > pivot_threshold)
                    local_min = std::min(local_min, (data.x_B[i] + harris_eps) / d[i]);
            }
            return local_min;
        },
        [](double a, double b) { return std::min(a, b); }
    );

    if (theta_min == std::numeric_limits<double>::max()) return -1; // задача необмежена

    // Прохід 2: серед рядків що задовольняють theta_min - обрати з найбільшим d[i]
    const auto best = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.m),
        std::pair<double, int>{ -1.0, -1 },
        // Обчислення6 ii = min((x_B)н / dн)
        [&](const tbb::blocked_range<int>& r, std::pair<double, int> local) {
            for (int i = r.begin(); i < r.end(); ++i) {
                if (d[i] > pivot_threshold &&
                    (data.x_B[i] + harris_eps) / d[i] <= theta_min &&
                    d[i] > local.first - pivot_threshold)
                {
                    local = { d[i], i };
                }
            }
            return local;
        },
        // Обчислення7 i_pivot = min(i_pivot, ii) (КД3)
        [](std::pair<double, int> a, std::pair<double, int> b) {
            if (a.second == -1) return b;
            if (b.second == -1) return a;
            return (a.first > b.first) ? a : b; // більший d[i] - чисельно стабільніший
        }
    );

    return best.second;
}

int ParallelSimplex::computeIPivotForDual() const {
    return tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.m),
        std::pair<double, int>{ -1e-9, -1 },
        [&](const tbb::blocked_range<int>& r, std::pair<double, int> local) {
            for (int i = r.begin(); i < r.end(); ++i) {
                if (data.x_B[i] < local.first)
                    local = { data.x_B[i], i };
            }
            return local;
        },
        // Обчислення7 i_pivot = min(i_pivot, ii) (КД3)
        [](std::pair<double, int> a, std::pair<double, int> b) {
            if (a.second == -1) return b;
            if (b.second == -1) return a;

            if (a.first < b.first) return a;
            if (b.first < a.first) return b;
            return (a.second < b.second) ? a : b;
        }
    ).second;
}

void ParallelSimplex::restoreCTB() {
    tbb::parallel_for(tbb::blocked_range<int>(0, data.m), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            const int idx = data.basisIdx[i];
            data.c_TB[i] = (idx < data.n) ? data.c[idx] : 0.0;
        }
    });
}
