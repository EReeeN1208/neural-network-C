cmake_minimum_required(VERSION 3.30)
project(c_neural_net C)

set(CMAKE_C_STANDARD 11)

add_executable(c_neural_net src/main.c
        src/linearalgebra.h
        src/linearalgebra.c
        src/tests.c
        src/activationFunctions.h
        src/activationFunctions.c
        src/tests.h
        src/csv.c
        src/csv.h
        src/mnist.c
        src/mnist.h
        src/util.c
        src/util.h
        src/nn.c
        src/nn.h)

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:${PROJECT_NAME}>/data
)