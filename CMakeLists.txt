cmake_minimum_required(VERSION 4.2)
project(diploma_hsc_for_linear_programming)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

add_executable(diploma_hsc_for_linear_programming main.cpp
        src/SharedData.cpp
        src/SimplexMath.h
        src/SimplexMethod.cpp
        src/ConsoleVisualizer.h
)

target_include_directories(diploma_hsc_for_linear_programming PRIVATE src)
target_link_libraries(diploma_hsc_for_linear_programming PRIVATE OpenMP::OpenMP_CXX TBB::tbb)
