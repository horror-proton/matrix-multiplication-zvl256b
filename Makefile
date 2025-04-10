CXX := riscv64-linux-gnu-g++

main: main.cpp solution.hpp
	${CXX} -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl -fopenmp -g -O2 -o $@ $< -std=gnu++2a -Wall -Wextra
