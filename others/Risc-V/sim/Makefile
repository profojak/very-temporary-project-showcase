ARCH=riscv64-none-elf

# C code
SOURCES = crt0local.S code.c
# Assembly
#SOURCES = assembly.S

TARGET_EXE = executable

CC=$(ARCH)-gcc
AS=$(ARCH)-as
LD=$(ARCH)-ld
OBJCOPY=$(ARCH)-objcopy
OBJDUMP=$(ARCH)-objdump

ARCHFLAGS += -mabi=ilp32
ARCHFLAGS += -march=rv32i
ARCHFLAGS += -fno-lto

CFLAGS  += -ggdb -Os -Wall
AFLAGS  += -ggdb

LDFLAGS += -ggdb
LDFLAGS += -nostartfiles
LDFLAGS += -nostdlib
LDFLAGS += -static

LDFLAGS += -Triscv32-bare.ld

LOADLIBES += -lgcc

CFLAGS  += $(ARCHFLAGS)
AFLAGS  += $(ARCHFLAGS)
LDFLAGS += $(ARCHFLAGS)

OBJECTS += $(filter %.o,$(SOURCES:%.S=%.o))
OBJECTS += $(filter %.o,$(SOURCES:%.c=%.o))
OBJECTS += $(filter %.o,$(SOURCES:%.cpp=%.o))

all: default

.PHONY : default clean dep all run

%.o:%.S
	$(CC) -D__ASSEMBLY__ $(AFLAGS) -c $< -o $@

%.o:%.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%.s:%.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -S $< -o $@

%.dat:%.bin
	hexdump -v -e '"%08x\n" ' $< >$@

%.bin:%
	$(OBJCOPY) --strip-debug -O binary $< $@

%.out:%
	$(OBJCOPY) -O ecoff-bigmips $< $@

default: $(TARGET_EXE).dat dump

dump: $(TARGET_EXE)
	$(OBJDUMP) -d -Mno-aliases $(TARGET_EXE) > $(TARGET_EXE).dump

$(TARGET_EXE): $(OBJECTS)
	$(CC) $(LDFLAGS) $^ $(LOADLIBES) -o $@

copy: default
	cp $(TARGET_EXE).dat ../instruction.hex
	cp $(TARGET_EXE).dump ../program.dump

clean:
	rm -f *.o *.a $(OBJECTS) $(TARGET_EXE) \
		$(TARGET_EXE).out $(TARGET_EXE).dat $(TARGET_EXE).dump
