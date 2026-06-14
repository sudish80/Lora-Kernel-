cmake_minimum_required(VERSION 3.18)
project(LoraKernel CXX)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -funroll-loops")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -arch=sm_80 --use_fast_math")

# Include directories
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR})

# ============================================================
# Optional GPU backends
# ============================================================
find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__CUDACC__ -DCUDA_ENABLED")
    enable_language(CUDA)
    set(CUDA_SOURCES)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
else()
    message(STATUS "CUDA not found - CPU-only mode")
endif()

find_package(hip QUIET)
if(hip_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")
    message(STATUS "ROCm found")
endif()

# Optional NCCL
find_package(NCCL QUIET)
if(NCCL_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNCCL_ENABLED")
    message(STATUS "NCCL found")
endif()

# Optional MKL
find_package(MKL QUIET)
if(MKL_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINTEL_MKL_VERSION")
    include_directories(${MKL_INCLUDE_DIR})
    list(APPEND EXTRA_LIBS ${MKL_LIBRARIES})
    message(STATUS "Intel MKL found")
endif()

# Required libraries
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# ============================================================
# All source files
# ============================================================
set(CORE_SOURCES
    src/core/signal_handler.cpp
)

set(NN_SOURCES
    src/nn/transformer_blocks.cpp
)

set(ALL_SOURCES
    main.cpp
    ${CORE_SOURCES}
    ${NN_SOURCES}
)

# ============================================================
# Main executable
# ============================================================
add_executable(lora_kernel ${ALL_SOURCES})

target_link_libraries(lora_kernel
    OpenMP::OpenMP_CXX
    OpenSSL::Crypto
    Threads::Threads
    ${EXTRA_LIBS}
)

# ============================================================
# Python bindings (pybind11)
# ============================================================
find_package(pybind11 QUIET CONFIG)
if(pybind11_FOUND)
    pybind11_add_module(lora_kernel_bindings bindings/python_module.cpp ${NN_SOURCES} ${CORE_SOURCES})
    target_link_libraries(lora_kernel_bindings PRIVATE OpenMP::OpenMP_CXX OpenSSL::Crypto Threads::Threads)
    install(TARGETS lora_kernel_bindings DESTINATION python)
    message(STATUS "pybind11 found - Python bindings enabled")
else()
    message(STATUS "pybind11 not found - Python bindings disabled")
endif()

# ============================================================
# Benchmark executable
# ============================================================
add_executable(benchmark tests/benchmark.cpp ${NN_SOURCES} ${CORE_SOURCES})
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX OpenSSL::Crypto Threads::Threads)

# ============================================================
# Test executables
# ============================================================
function(add_test_exec TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE} ${CORE_SOURCES} ${NN_SOURCES})
    target_link_libraries(${TEST_NAME}
        OpenMP::OpenMP_CXX
        OpenSSL::Crypto
        Threads::Threads
        ${EXTRA_LIBS}
    )
endfunction()

add_test_exec(test_gradient_checker tests/test_gradient_checker.cpp)
add_test_exec(test_convergence tests/test_convergence.cpp)
add_test_exec(test_mha tests/test_mha.cpp)
add_test_exec(test_kv_cache tests/test_kv_cache.cpp)
add_test_exec(test_e2e tests/test_e2e.cpp)
add_test_exec(test_nn tests/test_nn.cpp)
add_test_exec(test_backprop tests/test_backprop.cpp)
add_test_exec(test_distributed tests/test_distributed.cpp)

enable_testing()
add_test(NAME gradient_checker COMMAND test_gradient_checker)
add_test(NAME convergence COMMAND test_convergence)
add_test(NAME mha COMMAND test_mha)
add_test(NAME kv_cache COMMAND test_kv_cache)
add_test(NAME e2e COMMAND test_e2e)
add_test(NAME nn COMMAND test_nn)
add_test(NAME backprop COMMAND test_backprop)
add_test(NAME distributed COMMAND test_distributed)

install(TARGETS lora_kernel DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
