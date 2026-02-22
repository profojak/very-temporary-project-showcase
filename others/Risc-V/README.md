# 32-bit single-cycle RISC-V processor

> 32-bit single-cycle RISC-V processor.
> Hardware description for the architecture, memory components, and a comprehensive testbench.
> Implements a subset of RISC-V ISA, including load/store word, branching, and jump and link.
> Runs C!
> Beautiful (pls don't judge me) ASCII-art diagram.

Project for [Advanced Computer Architectures](https://bilakniha.cvut.cz/en/predmet4702206.html) course taught by [Pavel Píša](https://usermap.cvut.cz/profile/bc23926a-dd9a-4c16-bac3-cd6091d3c343), written in Verilog in 2024.

```
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                    32-bit single-cycle RISC-V processor                    //
//                                                                            //
//                                 Registers:                                 //
//               zero     0        hardwired to 0                             //
//               ra       1        return address                             //
//               sp       2        stack pointer                              //
//               gp       3        global pointer                             //
//               tp       4        thread pointer                             //
//               t0-2     5-7      temporary                                  //
//               s0/fp    8        saved/frame pointer                        //
//               s1       9        saved                                      //
//               a0-1     10-11    function arguments/return values           //
//               a2-7     12-17    function arguments                         //
//               s2-11    18-27    saved                                      //
//               t3-6     28-31    temporary                                  //
//                                                                            //
//                               Instructions:                                //
//               R-type (register)                                            //
//               31:25   24:20  19:15  14:12   11:7  6:0                      //
//               funct7  rs2    rs1    funct3  rd    op                       //
//               I-type (immediate)                                           //
//               31:20      19:15  14:12   11:7    6:0                        //
//               imm[11:0]  rs1    funct3  rd      op                         //
//               S-type (store)                                               //
//               31:25      24:20  19:15  14:12   11:7      6:0               //
//               imm[11:5]  rs2    rs1    funct3  imm[4:0]  op                //
//               B-type (branch)                                              //
//               31:25         24:20  19:15  14:12   11:7         6:0         //
//               imm[12,10:5]  rs2    rs1    funct3  imm[4:1,11]  op          //
//               U-type (upper immediate)                                     //
//               31:12       11:7  6:0                                        //
//               imm[31:12]  rd    op                                         //
//               J-type (jump)                                                //
//               31:12                  11:7  6:0                             //
//               imm[20,10:1,11,19:12]  rd    op                              //
//                                                                            //
//        add    R-type    0000000  XXXXX  XXXXX  000  XXXXX  0110011         //
//        sub    R-type    0100000  XXXXX  XXXXX  000  XXXXX  0110011         //
//        sll    R-type    0000000  XXXXX  XXXXX  001  XXXXX  0110011         //
//        slt    R-type    0000000  XXXXX  XXXXX  010  XXXXX  0110011         //
//        xor    R-type    0000000  XXXXX  XXXXX  100  XXXXX  0110011         //
//        srl    R-type    0000000  XXXXX  XXXXX  101  XXXXX  0110011         //
//         or    R-type    0000000  XXXXX  XXXXX  110  XXXXX  0110011         //
//        and    R-type    0000000  XXXXX  XXXXX  111  XXXXX  0110011         //
//       addi    I-type    IIIIIIIIIIII  XXXXX  000  XXXXX  0010011           //
//       xori    I-type    IIIIIIIIIIII  XXXXX  100  XXXXX  0010011           //
//        ori    I-type    IIIIIIIIIIII  XXXXX  110  XXXXX  0010011           //
//       andi    I-type    IIIIIIIIIIII  XXXXX  111  XXXXX  0010011           //
//         lw    I-type    IIIIIIIIIIII  XXXXX  010  XXXXX  0000011           //
//         sw    S-type    IIIIIII  XXXXX  XXXXX  010  IIIII  0100011         //
//        jal    J-type    IIIIIIIIIIIIIIIIIIII  XXXXX 1101111                //
//       jalr    I-type    IIIIIIIIIIII  XXXXX  000  XXXXX  1100111           //
//        beq    B-type    IIIIIII  XXXXX  XXXXX  000  XXXXX  1100011         //
//        bne    B-type    IIIIIII  XXXXX  XXXXX  001  XXXXX  1100011         //
//        bge    B-type    IIIIIII  XXXXX  XXXXX  101  XXXXX  1100011         //
//     ebreak    ebreak    0000000  00001  00000  000  00000  1110011         //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                                Control unit                                //
//                                                                            //
//                .---------.                                                 //
//  op[6:0] ---.--| Main    |-- _imm_select[1:0]                              //
//             |  | decoder |-- _alu_select                                   //
//             |  |         |-- _jump_select                                  //
//             |  |         |-- _result_select[0] -----o-----                 //
//             |  |         |-- _result_select[1] --o--'                      //
//             |  |         |-- _reg_write          |                         //
//             |  |         |-- _mem_write          '--\^^^\                  //
//             |  |         |-- _branch ------|^^^^\   | or |-- _pc_select    //
//             |  '---------'                 | and |--/___/                  //
//             |       |                   .--|____/                          //
//             |       | _alu_decode[1:0]  |                                  //
//             |       |                   '-----------------.                //
//  _zero ------------------------------------------\^^^^\   |                //
//             |       |                            | xor |--'                //
//             |  .------------.                 .--/____/                    //
//             '--| Arithmetic |                 |                            //
//  funct3[2:0] --| logic      |-- _branch_val --'                            //
//  funct7[6:0] --| decoder    |-- _alu_op[2:0]                               //
//                '------------'                                              //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                                 Data path                                  //
//                                                                            //
//                         .------ pc_plus_4 -------------------------.       //
//                         | .---- pc_target -----------------------. |       //
//                       .^^^^^.                                    | |       //
//  _pc_select ----------\ 0 1 /                                    | |       //
//                        ^^|^^                                     | |       //
//                          | pc_next                               | |       //
//                          |                                       | |       //
//                    .^^^^^^^^^^.                                  | |       //
//  _clock -----------| FlipFlop |                                  | |       //
//                    '-----.----'                4 --|^^^\         | |       //
//                          | pc                       > + |----------o       //
//                          o----------------------o--|___/         | |       //
//                          |                      |                | |       //
//                   .^^^^^^^^^^^^^.               '------------.   | |       //
//                   | Instruction |                            |   | |       //
//                   | memory      |                            |   | |       //
//                   '------.------'                            |   | |       //
//                          | instruction                       |   | |       //
//  Control unit <----o--o--o---- -----.                        |   | |       //
//                    |  |  |          |                        |   | |       //
//                    |  |  |    .--------------------------------. | |       //
//                    |  |  |    |     |                        | | | |       //
//                  .^^^^^^^^^^^^^^.   |                        | | | |       //
//  _clock ---------| Register     |   |                        | | | |       //
//  _reg_write -----| file         |   |      __                | | | |       //
//                  '--------------'   |__--^^  |               | | | |       //
//                    |          |   .^^        |               | | | |       //
//  _imm_select[1:0] ----------------| Extend   |               | | | |       //
//                    |          |   '--------.-'               | | | |       //
//                    |          |            | imm_extended    | | | |       //
//                    o---------------------. o---------------. | | | |       //
//                    |          |          | |               | | | | |       //
//                    |          |        .^^^^^.             | | | | |       //
//  _alu_select --------------------------\ 0 1 /             | | | | |       //
//                    |          |         ^^|^^              | | | | |       //
//                    |          | alu_in1   | alu_in2        | | | | |       //
//                    |          |           |                | | | | |       //
//                    |         \^^^^\   /^^^^/               | | | | |       //
//  _alu_op[2:0] ----------------\    \_/    /                | | | | |       //
//  Control unit <------- _zero --\   ALU   /                 | | | | |       //
//                    |            ^^^^|^^^^                  | | | | |       //
//                    | data_write     | alu_result    /^^^|--' | | | |       //
//                    |                o---o----. .---| + <     | | | |       //
//                    |                |   |    | |    \___|----' | | |       //
//                    |                |   |  .^^^^^.             | | |       //
//  _jump_select -----------------------------\ 0 1 /             | | |       //
//                    |                |   |   ^^|^^              | | |       //
//                    |                |   |     |                | | |       //
//                    |                |   |     '------------------' |       //
//                    |                |   |                      |   |       //
//                  .^^^^^^^^^^^^^^^^^^^.  |                      |   |       //
//  _clock ---------| Data              |  |                      |   |       //
//  _mem_write -----| memory            |  |                      |   |       //
//                  '-------------------'  |                      |   |       //
//                    | data_read          |                      |   |       //
//                    '-----------------.  |  .-----------------------'       //
//                                      |  |  |                   |           //
//                                    .^^^^^^^^^.                 |           //
//  _result_select[1:0] --------------\ 1  0  2 /                 |           //
//                                     ^^^^|^^^^                  |           //
//                                         | result               |           //
//                                         '----------------------'           //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
```
