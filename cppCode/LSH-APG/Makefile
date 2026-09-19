CXX = g++
CXXFLAGS = -std=c++17 -Ofast -march=native -fPIC -pthread -Wall -Wextra -I./include -I./src

LIB_SRCS = src/Metric.cpp src/LSBTree.cpp
LIB_OBJS = $(LIB_SRCS:.cpp=.o)

STATIC_LIB = liblsbtree.a
SHARED_LIB = liblsbtree.so
DEMO_EXEC = benchmark_driver

.PHONY: all clean

all: $(STATIC_LIB) $(SHARED_LIB) $(DEMO_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(STATIC_LIB): $(LIB_OBJS)
	ar rcs $@ $^

$(SHARED_LIB): $(LIB_OBJS)
	$(CXX) -shared -o $@ $^

$(DEMO_EXEC): examples/benchmark_driver.cpp $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $< -L. -llsbtree -o $@

clean:
	rm -f src/*.o examples/*.o $(STATIC_LIB) $(SHARED_LIB) $(DEMO_EXEC)