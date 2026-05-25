# diploma-hsc-for-linear-programming
Hardware and software complex for solving linear programming problems

## Build Instructions (Windows / MSYS2)

Для успішної компіляції та лінкування проєкту необхідно встановити бібліотеки OpenMP та Intel TBB через консоль MSYS2 MinGW64:

```bash
pacman -S mingw-w64-x86_64-openmp mingw-w64-x86_64-tbb ingw-w64-x86_64-cmake
```

When setting up your project in CLion (or another IDE), make sure you select Toolchain and CMake, which are located in your MSYS2 directory (`C:/msys64/mingw64`).
