CXXFLAGS = -std=c++20
CXX = g++

CDBG += -w
CXXFLAGS += $(CDBG) -v
CXXFLAGS += -g -O3


INCLUDE_PATH += -I$(HOME)
INCLUDE_PATH += -I$(HOME)/Documents/LoFloat/src

INCLUDE_PATH += -I$(HOME)/Documents/LoFloat/test


OSCILLATOR:
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) Oscillator.cpp -o oscillator

clean:
	rm -f oscillator
	rm -f *.o
	rm -rf *.dSYM
	rm -f *.exe
	rm -f *.out
	rm -f *.log
	rm -f *.txt
