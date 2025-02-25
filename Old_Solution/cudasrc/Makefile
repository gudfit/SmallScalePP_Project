all: build

build: 
	mkdir build && cd build && cmake .. && make && ./cuda_matmul > ../output.txt 2>&1

run:
	rm -f output.txt && cd build && make && ./cuda_matmul > ../output.txt 2>&1

clean:
	rm -f output.txt && rm -rf build

rebuild: clean build