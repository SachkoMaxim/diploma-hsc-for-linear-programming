#include "ParallelSimplex.h"

#include <limits>
#include <iostream>
#include <omp.h>
#include <oneapi/tbb.h>

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

#pragma omp critical
        // Введення даних
        switch (T_id)
        {
            // T1: Введення A
            case 0: data.putValuesIntoA_T(data.A_T, -20.0, 20.0, seed); break;
            // T2: Введення b, c, c_TB, x_B
            case 1:
                data.putValuesIntoB(data.b, 1.0, 50.0, seed);
                data.putValuesIntoVector(data.n, data.c, 1.0, 100.0, seed);
                std::fill(data.c_TB.begin(), data.c_TB.end(), 0.0);

                for (int i = 0; i < data.m; ++i) data.b[i] += 1e-11;
                break;
            // T3: Введення B
            case 2: data.putValuesIntoIdentityMatrix(data.m, data.m, data.B);    break;
            // T4: Введення B_m1
            case 3: data.putValuesIntoIdentityMatrix(data.m, data.m, data.B_m1); break;
            default: break;
        }

        // Бар'єр B1: синхронізація після введення даних усіма задачами
#pragma omp barrier

#pragma omp single
        {
            data.x_B = data.b;
            data.needsDualStart = false;
            for (int i = 0; i < data.m; ++i) {
                if (data.b[i] < 0.0) data.needsDualStart = true;
                data.constraintTypes[i] = ConstraintType::LESSEREQ;
            }
        }
    }

    std::cout << "Needs Dual Start: " << (data.needsDualStart ? "YES" : "NO") << "\n";

    iterations = 0;
    std::vector<double> d(data.m, 0.0);
    std::vector<double> u(data.n, 0.0);

    // Дуальна фаза: прибирання від'ємних x_B[i]
    if (data.needsDualStart) {
        std::vector<double> original_c = data.c;
        std::fill(data.c.begin(), data.c.end(), 0.0); // тимчасово нульова цільова функція

        while (iterations < maxIter) {
            ++iterations;

            i_pivot = computeIPivotForDual();
            if (i_pivot == -1) {
                std::cout << "Dual Phase: Feasible basis found in " << iterations << " iterations.\n";
                break;
            }

            // u = B_m1[i_pivot] * A
            data.saveRow(data.B_m1_iPiv, i_pivot);
            SimplexMath::MultiplyRowAndTransposedMatrix(data.B_m1_iPiv, data.A_T, u, data.m, data.n, P);

            // Обчислення1-4: v та j_pivot для двоїстого критерію
            arena.execute([&]() {
                computeVectorV();
                j_pivot = computeJPivot(
                    { std::numeric_limits<double>::max(), -1 },
                    [&](int j, double v_Aj) -> std::pair<bool, double> {
                        if (u[j] >= -1e-9) return { false, 0.0 };
                        return { true, std::abs((v_Aj - data.c[j]) / u[j]) };
                    }
                );
            });

            if (j_pivot == -1) {
                std::cout << "The problem is INFEASIBLE (No dual pivot column found).\n";
                return SimplexStatus::INFEASIBLE;
            }

            // Обчислення5: d = B^-1 * A[j_pivot]
            computeVectorD(d);

            // Обчислення8-12: оновлення базису
            performBasisUpdate(d, i_pivot, j_pivot);
        }

        data.c = original_c;

        // Відновлення c_TB після повернення оригінальної цільової функції
#pragma omp parallel for schedule(static) num_threads(P)
        for (int i = 0; i < data.m; ++i) {
            const int idx = data.basisIdx[i];
            data.c_TB[i] = (idx < data.n) ? data.c[idx] : 0.0;
        }
    }

    // Пряма фаза: рух до максимуму цільової функції
    while (iterations < maxIter) {
        ++iterations;

        // Обчислення1-4: v та j_pivot для прямого критерію
        arena.execute([&]() {
            computeVectorV();
            j_pivot = computeJPivot(
                { -1e-9, -1 },
                [&](int j, double v_Aj) -> std::pair<bool, double> {
                    const double delta = v_Aj - data.c[j];
                    return { delta < -1e-9, delta };
                }
            );
        });

        if (j_pivot == -1) {
            std::cout << "Optimal solution found in " << iterations << " iterations.\n";
            break;
        }

        // Обчислення5 d = B_m1 * A(j_pivot)н
        computeVectorD(d);

        // Обчислення6-7: вибір i_pivot за правилом Гарріса
        i_pivot = harrisRatioPivot(d);
        if (i_pivot == -1) {
            std::cout << "Target function is unbounded.\n";
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

    result = 0.0;
#pragma omp simd reduction(+:result)
    for (int j = 0; j < data.n; ++j) {
        result += data.c[j] * equationSolution[j];
    }

    return SimplexStatus::OPTIMAL;
}

void ParallelSimplex::computeVectorV() {
    v = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.m),
        std::vector<double>(data.m, 0.0),
        // Обчислення1: vi = c_TBн * B_m1н  (локальний внесок блоку рядків H)
        [&](const tbb::blocked_range<int> &r, std::vector<double> local_v) {
            for (int i = r.begin(); i != r.end(); ++i) {
                const double c_val = data.c_TB[i];
                if (std::abs(c_val) < 1e-12) continue;
                const double* row = data.B_m1.data() + i * data.m;
#pragma omp simd
                for (int j = 0; j < data.m; ++j) {
                    local_v[j] += c_val * row[j];
                }
            }
            return local_v;
        },
        // Обчислення2: v = v + vi (КД1)
        [](std::vector<double> a, const std::vector<double> &b) {
#pragma omp simd
            for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
            return a;
        }
    );
}

template<typename InnerFn>
int ParallelSimplex::computeJPivot(std::pair<double, int> identity, InnerFn&& innerFn) {
    // Спільна join-лямбда для обох варіантів
    //      .first  = чи є цей стовпець кандидатом
    //      .second = значення для порівняння (менше - краще)
    auto join = [](std::pair<double, int> a, std::pair<double, int> b) {
        if (a.second == -1) return b;
        if (b.second == -1) return a;
        if (a.first < b.first) return a;
        if (b.first < a.first) return b;
        return (a.second < b.second) ? a : b; // правило Бленда при рівних значеннях
    };

    return tbb::parallel_reduce(
        tbb::blocked_range<int>(0, data.n),
        identity,
        // Обчислення3: ji = min(v * Aн - cн)
        [&](const tbb::blocked_range<int> &r, std::pair<double, int> local) {
            for (int j = r.begin(); j < r.end(); ++j) {
                double v_Aj = 0.0;
                const double* col = data.A_T.data() + j * data.m;
#pragma omp simd reduction(+:v_Aj)
                for (int i = 0; i < data.m; ++i) {
                    v_Aj += v[i] * col[i];
                }

                auto [is_candidate, value] = innerFn(j, v_Aj);
                if (!is_candidate) continue;

                if (local.second == -1 || value < local.first) {
                    local = { value, j };
                }
                else if (std::abs(value - local.first) < 1e-11) {
                    if (j < local.second) {
                        local.second = j;
                    }
                }
            }
            return local;
        },
        join  // Обчислення4: j_pivot = min(j_pivot, ji) (КД2)
    ).second;
}

void ParallelSimplex::computeVectorD(std::vector<double> &d) {
    //Обчислення5 d = B_m1 * A(j_pivot)н
    arena.execute([&]() {
        tbb::parallel_for(tbb::blocked_range<int>(0, data.m),
            [&](const tbb::blocked_range<int> &r) {
                const double* col = data.A_T.data() + j_pivot * data.m;
                for (int i = r.begin(); i < r.end(); ++i) {
                    double sum = 0.0;
                    const double* row = data.B_m1.data() + i * data.m;
#pragma omp simd reduction(+:sum)
                    for (int j = 0; j < data.m; ++j) {
                        sum += row[j] * col[j];
                    }
                    d[i] = sum;
                }
            }
        );
    });
}

void ParallelSimplex::performBasisUpdate(const std::vector<double> &d, int ip, int jp) {
    arena.execute([&]() {
        tbb::parallel_invoke(
            // Обчислення8 c_TB(j_pivot) = c(j_pivot)
            [&]() { data.changeElement(data.c_TB, ip, jp); },
            // Обчислення9 d_ipRev = 1 / d(i_pivot)
            [&]() { data.d_ipRev = 1.0 / d[ip]; },
            // Копіювання x_B_iPiv = x_B(i_pivot) (КД4)
            [&]() { data.saveScalar(data.x_B_iPiv, data.x_B, ip); },
            // Копіювання B_m1_iPiv = B_m1(i_pivot) (КД5)
            [&]() { data.saveRow(data.B_m1_iPiv, ip); }
        );
    });

    // Обчислення10-12: оновлення B^-1 та x_B "на місці"
    SimplexMath::UpdateMatrixAndVectorParallel(
        data.B_m1_iPiv, d, data.d_ipRev,
        data.B_m1, data.x_B_iPiv, data.x_B,
        data.m, data.m, ip, P
    );
}

int ParallelSimplex::harrisRatioPivot(const std::vector<double> &d) const {
    constexpr double harris_eps = 1e-7;

    // Прохід 1: мінімально допустимий крок theta_min
    const double theta_min = arena.execute([&]() {
        return tbb::parallel_reduce(
            tbb::blocked_range<int>(0, data.m),
            std::numeric_limits<double>::max(),
            // Обчислення6 ii = min((x_B)н / dн)
            [&](const tbb::blocked_range<int>& r, double local_min) {
                for (int i = r.begin(); i < r.end(); ++i) {
                    if (d[i] > 1e-9)
                        local_min = std::min(local_min, (data.x_B[i] + harris_eps) / d[i]);
                }
                return local_min;
            },
            [](double a, double b) { return std::min(a, b); }
        );
    });

    if (theta_min == std::numeric_limits<double>::max()) return -1; // задача необмежена

    // Прохід 2: серед рядків що задовольняють theta_min - обрати з найбільшим d[i]
    const auto best = arena.execute([&]() {
        return tbb::parallel_reduce(
            tbb::blocked_range<int>(0, data.m),
            std::pair<double, int>{ -1.0, -1 },
            // Обчислення6 ii = min((x_B)н / dн)
            [&](const tbb::blocked_range<int>& r, std::pair<double, int> local) {
                for (int i = r.begin(); i < r.end(); ++i) {
                    if (d[i] > 1e-9 &&
                        data.x_B[i] / d[i] <= theta_min &&
                        d[i] > local.first - 1e-9)
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
    });

    return best.second;
}

#pragma omp declare reduction(min_loc : std::pair<double, int> : \
    omp_out = (omp_out.second == -1) ? omp_in : \
              (omp_in.second  == -1) ? omp_out : \
              (omp_in.first < omp_out.first) ? omp_in : \
              (omp_out.first < omp_in.first) ? omp_out : \
              (omp_in.second < omp_out.second) ? omp_in : omp_out) \
    initializer(omp_priv = { -1e-9, -1 })

int ParallelSimplex::computeIPivotForDual() {
    std::pair<double, int> global_min = { -1e-9, -1 };

    // Обчислення7 i_pivot = min(i_pivot, ii) (КД3)
#pragma omp parallel for reduction(min_loc: global_min) num_threads(P)
    for (int i = 0; i < data.m; ++i) {
        if (data.x_B[i] < global_min.first) {
            global_min = { data.x_B[i], i };
        }
    }

    return global_min.second;
}
