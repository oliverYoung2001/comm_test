.PHONY: clean
all: make_all

# codeGen:
# 	g++ -I../src sgd.cpp ../src/codegen.cpp ../src/dsl.cpp ../src/pipeline.cpp ../src/utils.cpp -g -O0 -o codeGen -std=c++14

make_all:
	${MAKE} -C csrc

clean:
	${MAKE} -C csrc clean