SRC_DIR		= ./src_cuda
APP_DIR		= ./apps

DEPS 		= $(SRC_DIR)/Pop.h $(SRC_DIR)/Prj.h $(SRC_DIR)/Globals.h $(SRC_DIR)/Pats.h $(SRC_DIR)/Parseparam.h $(SRC_DIR)/Logger.h
OBJS 		= $(SRC_DIR)/Pop.o $(SRC_DIR)/Prj.o $(SRC_DIR)/Globals.o $(SRC_DIR)/Pats.o $(SRC_DIR)/Parseparam.o $(SRC_DIR)/Logger.o

CC 			= 
CXX			= nvcc
MPICXX		= nvcc

INCLUDE		= -I$(SRC_DIR) # for header files

FLAGS		= -O3 -arch=sm_86 -std=c++11
CUDA_FLAGS	= -I$(SRC_DIR) -I/usr/local/include/ -I/opt/openmpi-4.0.7/include 
MPIXX_FLAGS	= -L/opt/local/lib/ -L/opt/openmpi-4.0.7/lib/ -lmpi -lcublas 

%.o: %.cpp $(DEPS)
	$(CXX) -x cu -c -o $@ $< $(INCLUDE) $(FLAGS) $(CUDA_FLAGS)

hidassospk: $(APP_DIR)/hidassospk/hidassospk.o $(OBJS)
	$(MPICXX) -o $(APP_DIR)/hidassospk/$@ $^ $(INCLUDE) $(FLAGS) $(MPIXX_FLAGS)

all: hidassospk

.PHONY: clean
clean : 
	rm -f *.o *.bin *.log *.png *.svg *.gif *.out out.txt err.txt *~ core reprlearnmain inpassomem hidassomemr deepff deepff2 deepff3 ammain deepassonet
	rm -f $(SRC_DIR)/*.o $(SRC_DIR)/*.bin $(SRC_DIR)/*~
	rm -f $(APP_DIR)/hidassospk/*.o $(APP_DIR)/hidassospk/*~ $(APP_DIR)/hidassospk/hidassospk