CXX = g++
CXXFLAGS = -fopenmp -Wall -O4
SRC_DIR = src
BUILD_DIR = matmul_perf

# Source files
SOURCES = $(SRC_DIR)/OpenMPSoll.cc $(SRC_DIR)/OpenMPSolMain.cc
OBJECTS = $(SOURCES:.cc=.o)
TARGET = $(BUILD_DIR)/matmul

# Default target
all: $(TARGET)

# Link the program
$(TARGET): $(OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(OBJECTS) $(CXXFLAGS) -o $(TARGET)

# Compile source files
%.o: %.cc $(SRC_DIR)/OpenMPSol.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean

