cmake_minimum_required(VERSION 4.2)
project(diploma_hsc_for_linear_programming)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

add_executable(diploma_hsc_for_linear_programming
        main.cpp
        src/SharedData.cpp
        src/SimplexMath.h
        src/ConsoleVisualizer.h
        src/ParallelSimplex.cpp
        src/FileWriter.h
)

target_include_directories(diploma_hsc_for_linear_programming PRIVATE src)

target_link_libraries(diploma_hsc_for_linear_programming PRIVATE
        OpenMP::OpenMP_CXX
        TBB::tbb
)

if(MSVC)
    target_compile_options(diploma_hsc_for_linear_programming PRIVATE /arch:AVX2 /fp:fast)
else()
    target_compile_options(diploma_hsc_for_linear_programming PRIVATE -march=native -ffast-math)
endif()
