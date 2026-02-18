# 32-bit single-cycle RISC-V processor

Circuit design using Verilog HDL, built with Verilator. Supported instructions
as well as the processor ASCII-art scheme are in the `riscv.v` source file.

## Building

To build and run a virtual simulation of the processor, run these commands:

```bash
make sim-build
make sim-run
```

The processor loads `data.hex` and `instruction.hex` files into its memory and
executes the program until it reaches `ebreak` or ends after a certain delay.
The human-readable code is stored in `program.dump`. The simulation outputs
to terminal whenever `sw` instruction is executed.

The following commands build and invoke processor unittest testbench:

```bash
make test-build
make test-run
```

To check the quality of the code with Verilator lint, run `make lint`. To clean
up, run `make clean`.

## Coding

One can write its own program in assembly or C language and compile it either
with `make compile`, which also copies the compiled program to `instruction.hex`
and `program.dump` files, or go to the `sim/` directory and run `make` there.
This compiles the program locally. To copy it to the processor files, run
`make copy` from the `sim/` directory.

The `assembly.S` file contains the assembly code with all the supported
instructions. The comments in the file state the expected output when run with
the following contents of `data.hex`:

```
00000000
00000001
00000004
```

The `code.c` file contains C code of *greatest common divisor* algorithm.
To choose between the assembly and C code, change the `SOURCES` in Makefile.
To change the input data, modify the `data.hex` file.
