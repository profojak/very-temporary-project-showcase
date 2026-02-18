
start:
_start:
  # For input 1, 4
  lw t0, 4(zero)
  lw t1, 8(zero)
  add t2, t0, t1
  sw t2, 12(zero) # 5
  sub t2, t1, t0
  sw t2, 16(zero) # 3
  sll t2, t1, t0
  sw t2, 20(zero) # 8
  slt t2, t0, t1
  sw t2, 24(zero) # 1
  xor t2, t0, t1
  sw t2, 28(zero) # 5
  srl t2, t1, t0
  sw t2, 32(zero) # 2
  or t2, t0, t1
  sw t2, 36(zero) # 5
  and t2, t0, t1
  sw t2, 40(zero) # 0
  addi t3, zero, 64
  sw t3, 44(zero) # 64
  xori t2, t1, 5
  sw t2, 48(zero) # 1
  ori t2, t1, 5
  sw t2, 52(zero) # 5
  andi t2, t1, 5
  sw t2, 56(zero) # 4
  jal ra, jal
  sw t0, 60(zero) # 1 - Unreachable
  ebreak

jal:
  sw t3, 60(zero) # 64
  addi t4, zero, 0x288
  jalr ra, t4, 0
  sw t0, 64(zero) # 1 - Unreachable
  ebreak

jalr:
  sw t3, 64(zero) # 64
  beq t0, t0, beq
  sw t0, 68(zero) # 1 - Unreachable
  ebreak

beq:
  sw t3, 68(zero) # 64
  bne t1, t0, bne
  sw t0, 72(zero) # 1 - Unreachable
  ebreak

bne:
  sw t3, 72(zero) # 64
  bge t1, t0, bge
  sw t0, 76(zero) # 1 - Unreachable
  ebreak

bge:
  sw t3, 76(zero) # 64
  ebreak

data:
  .word 00000001
  .word 00000004

.end start
