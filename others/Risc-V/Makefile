WARNINGS = -Wall -Wno-DECLFILENAME -Wno-MULTITOP -Wno-UNUSEDSIGNAL

all: help

help:
	@echo "make lint       - Run linting"
	@echo "make sim-build  - Build simulation"
	@echo "make sim-run    - Run simulation"
	@echo "make test-build - Build unittest testbench"
	@echo "make test-run   - Run unittest testbench"
	@echo "make compile    - Compile code for simulation"
	@echo "make clean      - Clean up"

lint:
	verilator --lint-only --timing $(WARNINGS) riscv.v

sim-build:
	verilator --binary --trace --top Simulation riscv.v || true
	make -j -C obj_dir -f VSimulation.mk VSimulation

sim-run:
	obj_dir/VSimulation

test-build:
	verilator --binary --trace --top TestBench riscv.v || true
	make -j -C obj_dir -f VTestBench.mk VTestBench

test-run:
	obj_dir/VTestBench

compile:
	make -C sim
	make -C sim copy

clean:
	rm -rf obj_dir
	make -C sim clean
