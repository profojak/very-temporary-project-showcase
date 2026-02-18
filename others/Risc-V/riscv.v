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


`timescale 1ns/1ps


module RiscV(  // Processor
  input _clock,
  input _reset
);

  wire _pc_select;
  wire [1:0] _imm_select;
  wire _alu_select;
  wire _jump_select;
  wire [1:0] _result_select;
  wire _reg_write;
  wire _mem_write;
  wire [2:0] _alu_op;
  wire _zero;

  wire [31:0] instruction;
  wire [31:0] data_read;
  wire [31:0] data_write;
  wire [31:0] alu_result;
  wire [31:0] pc;

  Control CONTROL(
    ._zero(_zero),
    ._pc_select(_pc_select),
    ._imm_select(_imm_select),
    ._alu_select(_alu_select),
    ._jump_select(_jump_select),
    ._result_select(_result_select),
    ._reg_write(_reg_write),
    ._mem_write(_mem_write),
    ._alu_op(_alu_op),
    .op(instruction[6:0]),
    .funct3(instruction[14:12]),
    .funct7(instruction[31:25])
  );

  DataPath DATA_PATH(
    ._clock(_clock),
    ._reset(_reset),
    ._pc_select(_pc_select),
    ._imm_select(_imm_select),
    ._alu_select(_alu_select),
    ._jump_select(_jump_select),
    ._result_select(_result_select),
    ._reg_write(_reg_write),
    ._alu_op(_alu_op),
    ._zero(_zero),
    .instruction(instruction),
    .data_read(data_read),
    .data_write(data_write),
    .alu_result(alu_result),
    .pc(pc)
  );

  DataMem DATA_MEM(
    ._clock(_clock),
    ._write_enable(_mem_write),
    .addr(alu_result),
    .in(data_write),
    .out(data_read)
  );

  InstructionMem INSTRUCTION_MEM(
    .addr(pc),
    .out(instruction)
  );
endmodule


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


module Control(  // Control unit
  input _zero,
  output _pc_select,
  output [1:0] _imm_select,
  output _alu_select,
  output _jump_select,
  output [1:0] _result_select,
  output _reg_write,
  output _mem_write,
  output [2:0] _alu_op,

  input [6:0] op,
  input [2:0] funct3,
  input [6:0] funct7
);

  wire _branch;
  wire _branch_val;
  wire _branch_xor;
  wire [1:0] _alu_decode;

  assign _branch_xor = _zero ^ _branch_val;
  assign _pc_select = _result_select[1] | _branch & _branch_xor;

  MainDecode MAIN_DECODE(
    ._imm_select(_imm_select),
    ._alu_select(_alu_select),
    ._jump_select(_jump_select),
    ._result_select(_result_select),
    ._reg_write(_reg_write),
    ._mem_write(_mem_write),
    ._branch(_branch),
    ._alu_decode(_alu_decode),
    .op(op)
  );

  ArithmeticLogicDecode ALU_DECODE(
    ._alu_decode(_alu_decode),
    ._alu_op(_alu_op),
    ._branch_val(_branch_val),
    .op(op),
    .funct3(funct3),
    .funct7(funct7)
  );
endmodule


module MainDecode(  // Main decoder
  output reg [1:0] _imm_select,
  output reg _alu_select,
  output reg _jump_select,
  output reg [1:0] _result_select,
  output reg _reg_write,
  output reg _mem_write,
  output reg _branch,
  output reg [1:0] _alu_decode,

  input [6:0] op
);

  always @(*)
    case (op)
      // R-type
      7'b0110011: begin
        _imm_select = 2'b00;
        _alu_select = 0;
        _jump_select = 0;
        _result_select = 2'b00;
        _reg_write = 1;
        _mem_write = 0;
        _branch = 0;
        _alu_decode = 0'b00;
        //$display("R-type");
      end
      // I-type
      7'b0010011: begin
        _imm_select = 2'b00;
        _alu_select = 1;
        _jump_select = 0;
        _result_select = 2'b00;
        _reg_write = 1;
        _mem_write = 0;
        _branch = 0;
        _alu_decode = 0'b00;
        //$display("I-type");
      end
      // lw
      7'b0000011: begin
        _imm_select = 2'b00;
        _alu_select = 1;
        _jump_select = 0;
        _result_select = 2'b01;
        _reg_write = 1;
        _mem_write = 0;
        _branch = 0;
        _alu_decode = 0'b01;
        //$display("lw");
      end
      // sw
      7'b0100011: begin
        _imm_select = 2'b01;
        _alu_select = 1;
        _jump_select = 0;
        _result_select = 2'b00;
        _reg_write = 0;
        _mem_write = 1;
        _branch = 0;
        _alu_decode = 0'b01;
        //$display("sw");
      end
      // jal
      7'b1101111: begin
        _imm_select = 2'b11;
        _alu_select = 1;
        _jump_select = 0;
        _result_select = 2'b10;
        _reg_write = 1;
        _mem_write = 0;
        _branch = 0;
        _alu_decode = 0'b00;
        //$display("jal");
      end
      // jalr
      7'b1100111: begin
        _imm_select = 2'b11;
        _alu_select = 1;
        _jump_select = 1;
        _result_select = 2'b10;
        _reg_write = 1;
        _mem_write = 0;
        _branch = 0;
        _alu_decode = 0'b00;
        //$display("jalr");
      end
      // B-type
      7'b1100011: begin
        _imm_select = 2'b10;
        _alu_select = 0;
        _jump_select = 0;
        _result_select = 2'b00;
        _reg_write = 0;
        _mem_write = 0;
        _branch = 1;
        _alu_decode = 0'b10;
        //$display("B-type");
      end
      default: begin
        _imm_select = 2'b00;
        _alu_select = 0;
        _jump_select = 0;
        _result_select = 2'b00;
        _reg_write = 0;
        _mem_write = 0;
        _branch = 0;
        _alu_decode = 0;
        // ebreak
        if (op == 7'b1110011) begin
          $display("ebreak");
          $finish;
        end
      end
    endcase
endmodule


module ArithmeticLogicDecode(  // Arithmetic logic unit decoder
  input [1:0] _alu_decode,
  output reg [2:0] _alu_op,
  output reg _branch_val,

  input [6:0] op,
  input [2:0] funct3,
  input [6:0] funct7
);

  wire _sub_instruction;

  assign _sub_instruction = op[5] & funct7[5];

  always @(*)
    case (_alu_decode)
      0'b00: if (_sub_instruction) _alu_op = 3'b011;
             else _alu_op = funct3;
      // lw, sw
      0'b01: _alu_op = 3'b000;
      // B-type
      0'b10: case (funct3)
        // bne
        0'b001: begin
          _alu_op = 3'b011;
          _branch_val = 1;
        end
        // bge
        0'b101: begin
          _alu_op = 3'b010;
          _branch_val = 0;
        end
        default: begin
          _alu_op = 3'b011;
          _branch_val = 0;
        end
      endcase
    endcase
endmodule


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


module DataPath(  // Data path
  input _clock,
  input _reset,
  input _pc_select,
  input [1:0] _imm_select,
  input _alu_select,
  input _jump_select,
  input [1:0] _result_select,
  input _reg_write,
  input [2:0] _alu_op,
  output _zero,

  input [31:0] instruction,
  input [31:0] data_read,
  output [31:0] data_write,
  output [31:0] alu_result,
  output [31:0] pc
);

  // Program counter

  wire [31:0] pc_next;
  wire [31:0] pc_plus_4;
  wire [31:0] pc_imm;
  wire [31:0] pc_target;

  FlipFlop PC_REG(
    ._clock(_clock),
    ._reset(_reset),
    .in(pc_next),
    .out(pc)
  );

  assign pc_plus_4 = pc + 4;
  assign pc_imm = pc + imm_extended;
  assign pc_target = _jump_select ? alu_result : pc_imm;
  assign pc_next = _pc_select ? pc_target : pc_plus_4;

  // Register file

  wire [31:0] result;

  RegFile REG_FILE(
    ._clock(_clock),
    ._write_enable(_reg_write),
    .addr1(instruction[19:15]),
    .addr2(instruction[24:20]),
    .addr3(instruction[11:7]),
    .in(result),
    .out1(alu_in1),
    .out2(data_write)
  );

  wire [31:0] imm_extended;

  Extend EXTEND(
    ._imm_select(_imm_select),
    .instruction(instruction[31:7]),
    .imm_extended(imm_extended)
  );

  // Arithmetic logic unit

  wire [31:0] alu_in1;
  wire [31:0] alu_in2;

  assign alu_in2 = _alu_select ? imm_extended : data_write;

  ArithmeticLogic ALU(
    ._op(_alu_op),
    ._zero(_zero),
    .in1(alu_in1),
    .in2(alu_in2),
    .out(alu_result)
  );

  assign result = _result_select[1] ? pc_plus_4 :
    _result_select[0] ? data_read : alu_result;
endmodule


module Extend(  // Sign extension unit
  input [1:0] _imm_select,

  input [31:7] instruction,
  output reg [31:0] imm_extended
);

  always @(*)
    case (_imm_select)
      // I-type
      2'b00: imm_extended = {{20{instruction[31]}}, instruction[31:20]};
      // S-type
      2'b01: imm_extended = {{20{instruction[31]}}, instruction[31:25],
                             instruction[11:7]};
      // B-type
      2'b10: imm_extended = {{20{instruction[31]}}, instruction[7],
                             instruction[30:25], instruction[11:8], 1'b0};
      // J-type
      2'b11: imm_extended = {{12{instruction[31]}}, instruction[19:12],
                             instruction[20], instruction[30:21], 1'b0};
      // No U-type instruction implemented
      default: imm_extended = 0;
    endcase
endmodule


module ArithmeticLogic(  // Arithmetic logic unit
  input [2:0] _op,
  output _zero,

  input [31:0] in1,
  input [31:0] in2,
  output reg [31:0] out
);

  always @(*)
    case (_op[2:0])
      0'b000: out = in1 + in2;
      0'b001: out = in1 << in2;
      0'b011: out = in1 - in2;
      0'b010: out = in1 < in2 ? 1 : 0;
      0'b100: out = in1 ^ in2;
      0'b101: out = in1 >> in2;
      0'b110: out = in1 | in2;
      0'b111: out = in1 & in2;
      default: out = 0;
    endcase

  assign _zero = (out == 0);
endmodule


////////////////////////////////////////////////////////////////////////////////


module FlipFlop(  // Flip flop
  input _clock,
  input _reset,

  input [31:0] in,
  output reg [31:0] out
);

  always @(posedge _clock, posedge _reset)
    if (_reset) out <= 32'h200;
    else out <= in;
endmodule


module RegFile(  // Three port register file with 32 registers
  input _clock,
  input _write_enable,

  input [4:0] addr1,
  input [4:0] addr2,
  input [4:0] addr3,
  input [31:0] in,
  output [31:0] out1,
  output [31:0] out2
);

  reg [31:0] REGS [31:0];

  // Register 0 hardwired to 0
  assign out1 = (addr1 == 0) ? 0 : REGS[addr1];
  assign out2 = (addr2 == 0) ? 0 : REGS[addr2];

  always @(posedge _clock) if (_write_enable) REGS[addr3] <= in;
endmodule


module DataMem(  // Data memory with 128 words
  input _clock,
  input _write_enable,

  input [31:0] addr,
  input [31:0] in,
  output [31:0] out
);

  reg [31:0] MEM [0:1023];

  initial $readmemh("data.hex", MEM);

  // Word-aligned access
  assign out = MEM[addr[11:2]];

  always @(posedge _clock)
    if (_write_enable) begin
      MEM[addr[11:2]] <= in;
      $display("  %h: %5d", addr, in);
    end
endmodule


module InstructionMem(  // Instruction memory with 256 words
  input [31:0] addr,
  output [31:0] out
);

  reg [31:0] MEM [0:511];

  initial $readmemh("instruction.hex", MEM);

  // Word-aligned access
  assign out = MEM[addr[10:2]];
endmodule


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                              Module testbench                              //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


module TestBench;  // Testbench
  MainDecode_TB MAIN_DECODE_TB();
  ArithmeticLogicDecode_TB ALU_DECODE_TB();

  Extend_TB EXTEND_TB();

  FlipFlop_TB FLIP_FLOP_TB();
  RegFile_TB REG_FILE_TB();

  initial #1000 $finish;
endmodule


module MainDecode_TB;  // Main decoder testbench

  wire [1:0] _imm_select;
  wire _alu_select;
  wire _jump_select;
  wire [1:0] _result_select;
  wire _reg_write;
  wire _mem_write;
  wire _branch;
  wire [1:0] _alu_decode;

  reg [6:0] op;

  MainDecode MAIN_DECODE(
    ._imm_select(_imm_select),
    ._alu_select(_alu_select),
    ._jump_select(_jump_select),
    ._result_select(_result_select),
    ._reg_write(_reg_write),
    ._mem_write(_mem_write),
    ._branch(_branch),
    ._alu_decode(_alu_decode),
    .op(op)
  );

  initial begin
    // R-type
    #1 op = 7'b0110011;
    #1 if (_alu_select !== 0 ||
           _jump_select !== 0 ||
           _result_select !== 2'b00 ||
           _reg_write !== 1 ||
           _mem_write !== 0 ||
           _branch !== 0 ||
           _alu_decode !== 0'b00)
      $display("FAIL! MainDec: R-type");
    // I-type
    #1 op = 7'b0010011;
    #1 if (_imm_select !== 2'b00 ||
           _alu_select !== 1 ||
           _jump_select !== 0 ||
           _result_select !== 2'b00 ||
           _reg_write !== 1 ||
           _mem_write !== 0 ||
           _branch !== 0 ||
           _alu_decode !== 0'b00)
      $display("FAIL! MainDec: I-type");
    // lw
    #1 op = 7'b0000011;
    #1 if (_imm_select !== 2'b00 ||
           _alu_select !== 1 ||
           _jump_select !== 0 ||
           _result_select !== 2'b01 ||
           _reg_write !== 1 ||
           _mem_write !== 0 ||
           _branch !== 0 ||
           _alu_decode !== 0'b01)
      $display("FAIL! MainDec: lw");
    // sw
    #1 op = 7'b0100011;
    #1 if (_imm_select !== 2'b01 ||
           _alu_select !== 1 ||
           _jump_select !== 0 ||
           _result_select !== 2'b00 ||
           _reg_write !== 0 ||
           _mem_write !== 1 ||
           _branch !== 0 ||
           _alu_decode !== 0'b01)
      $display("FAIL! MainDec: sw");
    // jal
    #1 op = 7'b1101111;
    #1 if (_imm_select !== 2'b11 ||
           _alu_select !== 1 ||
           _jump_select !== 0 ||
           _result_select !== 2'b10 ||
           _reg_write !== 1 ||
           _mem_write !== 0 ||
           _branch !== 0 ||
           _alu_decode !== 0'b00)
      $display("FAIL! MainDec: jal");
    // jalr
    #1 op = 7'b1100111;
    #1 if (_imm_select !== 2'b11 ||
           _alu_select !== 1 ||
           _jump_select !== 1 ||
           _result_select !== 2'b10 ||
           _reg_write !== 1 ||
           _mem_write !== 0 ||
           _branch !== 0 ||
           _alu_decode !== 0'b00)
      $display("FAIL! MainDec: jalr");
    // B-type
    #1 op = 7'b1100011;
    #1 if (_imm_select !== 2'b10 ||
           _alu_select !== 0 ||
           _jump_select !== 0 ||
           _result_select !== 2'b00 ||
           _reg_write !== 0 ||
           _mem_write !== 0 ||
           _branch !== 1 ||
           _alu_decode !== 0'b10)
      $display("FAIL! MainDec: B-type");
  end
endmodule


module ArithmeticLogicDecode_TB;  // Arithmetic logic unit decoder testbench

  reg [1:0] _alu_decode;
  wire [2:0] _alu_op;
  wire _branch_val;

  reg [6:0] op;
  reg [2:0] funct3;
  reg [6:0] funct7;

  ArithmeticLogicDecode ALU_DECODE(
    ._alu_decode(_alu_decode),
    ._alu_op(_alu_op),
    ._branch_val(_branch_val),
    .op(op),
    .funct3(funct3),
    .funct7(funct7)
  );

  initial begin
    _alu_decode = 2'b00;
    funct7 = 7'b0000000;
    op = 7'b0110011;
    // add
    #1 funct3 = 3'b000;
    #1 if (_alu_op !== 3'b000)
      $display("FAIL! ALUDec: _alu_op = %b, expected 000", _alu_op);
    // sub
    #1 funct7 = 7'b0100000;
    #1 if (_alu_op !== 3'b011)
      $display("FAIL! ALUDec: _alu_op = %b, expected 011", _alu_op);
    // sll
    #1 begin
      funct7 = 7'b0000000;
      funct3 = 3'b001;
    end
    #1 if (_alu_op !== 3'b001)
      $display("FAIL! ALUDec: _alu_op = %b, expected 001", _alu_op);
    // slt
    #1 funct3 = 3'b010;
    #1 if (_alu_op !== 3'b010)
      $display("FAIL! ALUDec: _alu_op = %b, expected 010", _alu_op);
    // xor
    #1 funct3 = 3'b100;
    #1 if (_alu_op !== 3'b100)
      $display("FAIL! ALUDec: _alu_op = %b, expected 100", _alu_op);
    // srl
    #1 funct3 = 3'b101;
    #1 if (_alu_op !== 3'b101)
      $display("FAIL! ALUDec: _alu_op = %b, expected 101", _alu_op);
    // or
    #1 funct3 = 3'b110;
    #1 if (_alu_op !== 3'b110)
      $display("FAIL! ALUDec: _alu_op = %b, expected 110", _alu_op);
    // and
    #1 funct3 = 3'b111;
    #1 if (_alu_op !== 3'b111)
      $display("FAIL! ALUDec: _alu_op = %b, expected 111", _alu_op);
    // addi
    #1 begin
      op = 7'b0010011;
      funct3 = 3'b000;
    end
    #1 if (_alu_op !== 3'b000)
      $display("FAIL! ALUDec: _alu_op = %b, expected 000", _alu_op);
    // xori
    #1 funct3 = 3'b100;
    #1 if (_alu_op !== 3'b100)
      $display("FAIL! ALUDec: _alu_op = %b, expected 100", _alu_op);
    // ori
    #1 funct3 = 3'b110;
    #1 if (_alu_op !== 3'b110)
      $display("FAIL! ALUDec: _alu_op = %b, expected 110", _alu_op);
    // andi
    #1 funct3 = 3'b111;
    #1 if (_alu_op !== 3'b111)
      $display("FAIL! ALUDec: _alu_op = %b, expected 111", _alu_op);
    // lw
    #1 begin
      _alu_decode = 2'b01;
      op = 7'b0000011;
      funct3 = 3'b010;
    end
    #1 if (_alu_op !== 3'b000)
      $display("FAIL! ALUDec: _alu_op = %b, expected 000", _alu_op);
    // sw
    #1 begin
      _alu_decode = 2'b01;
      op = 7'b0100011;
      funct3 = 3'b010;
    end
    #1 if (_alu_op !== 3'b000)
      $display("FAIL! ALUDec: _alu_op = %b, expected 000", _alu_op);
    // beq
    #1 begin
      _alu_decode = 2'b10;
      op = 7'b1100011;
      funct3 = 3'b000;
    end
    #1 if (_alu_op !== 3'b011)
      $display("FAIL! ALUDec: _alu_op = %b, expected 011", _alu_op);
    // bne
    #1 begin
      _alu_decode = 2'b10;
      op = 7'b1100011;
      funct3 = 3'b001;
    end
    #1 if (_alu_op !== 3'b011)
      $display("FAIL! ALUDec: _alu_op = %b, expected 011", _alu_op);
    // bge
    #1 begin
      _alu_decode = 2'b10;
      op = 7'b1100011;
      funct3 = 3'b101;
    end
    #1 if (_alu_op !== 3'b010)
      $display("FAIL! ALUDec: _alu_op = %b, expected 010", _alu_op);
  end
endmodule


////////////////////////////////////////////////////////////////////////////////


module Extend_TB;  // Sign extension unit testbench

  reg [1:0] _imm_select;

  reg [31:0] instruction;
  wire [31:0] imm_extended;

  Extend EXTEND(
    ._imm_select(_imm_select),
    .instruction(instruction[31:7]),
    .imm_extended(imm_extended)
  );

  initial begin
    // I-type
    #1 begin
      _imm_select = 2'b00;
      instruction = 32'h8f100000;
    end
    #1 if (imm_extended !== 32'hfffff8f1)
      $display("FAIL! Extend: imm_extended = %h, expected fffff8f1",
        imm_extended);
    // S-type
    #1 begin
      _imm_select = 2'b01;
      instruction = 32'hfc000f00;
    end
    #1 if (imm_extended !== 32'hffffffde)
      $display("FAIL! Extend: imm_extended = %h, expected ffffffde",
        imm_extended);
    // B-type
    #1 begin
      _imm_select = 2'b10;
      instruction = 32'h7c000880;
    end
    #1 if (imm_extended !== 32'h00000fd0)
      $display("FAIL! Extend: imm_extended = %h, expected 00000fd0",
        imm_extended);
    // J-type
    #1 begin
      _imm_select = 2'b11;
      instruction = 32'h3ff7f000;
    end
    #1 if (imm_extended !== 32'h0007fbfe)
      $display("FAIL! Extend: imm_extended = %h, expected 0007fbfe",
        imm_extended);
  end
endmodule


////////////////////////////////////////////////////////////////////////////////


module FlipFlop_TB;  // Flip flop testbench

  reg _clock = 0;
  reg _reset = 0;
  reg [31:0] in = 0;
  wire [31:0] out;

  FlipFlop FLIP_FLOP(
    ._clock(_clock),
    ._reset(_reset),
    .in(in),
    .out(out)
  );

  always #5 _clock <= !_clock;

  initial begin
    #10 if (out !== 0)
      $display("FAIL! FlipFlop: out = %d, expected 0", out);
    #5 in = 1;
    #10 if (out !== 1)
      $display("FAIL! FlipFlop: out = %d, expected 1", out);
    #5 in = 2;
    #10 if (out !== 2)
      $display("FAIL! FlipFlop: out = %d, expected 2", out);
    #5 in = 3; _reset = 1;
    #10 if (out !== 512)
      $display("FAIL! FlipFlop: out = %d, expected 512", out);
    #5 in = 4;
    #10 if (out !== 512)
      $display("FAIL! FlipFlop: out = %d, expected 512", out);
    #5 _reset = 0;
    #10 if (out !== 4)
      $display("FAIL! FlipFlop: out = %d, expected 4", out);
  end
endmodule


module RegFile_TB;  // Register file testbench

  reg _clock = 0;
  reg _write_enable = 0;
  reg [4:0] addr1 = 0;
  reg [4:0] addr2 = 0;
  reg [4:0] addr3 = 0;
  reg [31:0] in = 0;
  wire [31:0] out1;
  wire [31:0] out2;

  RegFile REG_FILE(
    ._clock(_clock),
    ._write_enable(_write_enable),
    .addr1(addr1),
    .addr2(addr2),
    .addr3(addr3),
    .in(in),
    .out1(out1),
    .out2(out2)
  );

  always #5 _clock <= !_clock;

  initial begin
    #5 _write_enable = 1; addr3 = 0; in = 1;
    #10 addr3 = 1; in = 2;
    #10 addr3 = 2; in = 3;
    #10 addr3 = 3; in = 4;
    #10 addr3 = 31; in = 5;
    #10 _write_enable = 0; addr1 = 0; addr2 = 1;
    #5 begin
      if (out1 !== 0)
        $display("FAIL! RegFile: out1 = %d, expected 0", out1);
      if (out2 !== 2)
        $display("FAIL! RegFile: out2 = %d, expected 2", out2);
    end
    #5 addr1 = 2; addr2 = 3;
    #5 begin
      if (out1 !== 3)
        $display("FAIL! RegFile: out1 = %d, expected 3", out1);
      if (out2 !== 4)
        $display("FAIL! RegFile: out2 = %d, expected 4", out2);
    end
    #5 addr1 = 31; addr2 = 31;
    #5 begin
      if (out1 !== 5)
        $display("FAIL! RegFile: out1 = %d, expected 5", out1);
      if (out2 !== 5)
        $display("FAIL! RegFile: out2 = %d, expected 5", out2);
    end
  end
endmodule


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                                 Simulation                                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


module Simulation;

  reg _clock = 0;
  reg _reset = 0;

  RiscV RISC_V(
    ._clock(_clock),
    ._reset(_reset)
  );

  initial begin
    $display("Simulation output:");
    _reset = 1;
    #10 _reset = 0;
    #10000 $finish;
  end

  always #5 _clock <= !_clock;
endmodule


////////////////////////////////////////////////////////////////////////////////
