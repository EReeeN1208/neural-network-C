cmake_minimum_required(VERSION 3.30)
project(c_neural_net C)

set(CMAKE_C_STANDARD 11)

add_executable(c_neural_net src/main.c
        src/linearalgebra.h
        src/linearalgebra.c)
