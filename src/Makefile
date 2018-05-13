include ../Makefile.common

SRC = AcqConfig.cpp Algo.cpp Blob.cpp Grid.cpp LinOp.cpp BlobImage.cpp	\
BlobInterpl.cpp BlobProjector.cpp Projector.cpp Tools.cpp SpLinOp.cpp	\
LinSolver.cpp BlobImageTools.cpp
OBJ = $(addprefix $(OBJ_T2D)/, $(SRC:.cpp=.o))
#OBJ = $(subst .cpp,.o, $(SRC))

SRC_CU = BlobInterpl.GPU.cu BlobProjector_NormEstimator.GPU.cu		\
BlobProjector.GPU.cu PixDrvProjector.GPU.cu # BlobMatching.GPU.cu	\
#MARTOp.GPU.cu
OBJ_CU = $(addprefix $(OBJ_T2D)/, $(SRC_CU:.cu=.o))
#OBJ_CU = $(SRC_CU:.cu=.o)

SPALGO_SRC_TMP = SpAlgo.cpp TVADM.cpp TVAL3.cpp TVL1Prxm.cpp	\
TVL1SPL.cpp L1FISTA.cpp L1FISTA_Homotopy.cpp L1PADM.cpp		\
MatchingPursuit.cpp CoSaMP.cpp L0IHT.cpp TVL1Prxm_Homotopy.cpp
# L1DADM.cpp #TVIADM.cpp  TVAL3_IADM.cpp # TVGradient.cpp TVIADM.cpp EdgeTV.cpp AdaptL1.cpp EdgeTVL1.cpp
SPALGO_SRC = $(addprefix SpAlgo/, $(SPALGO_SRC_TMP:.cpp=.o))
SPALGO_OBJ = $(addprefix $(OBJ_T2D)/, $(SPALGO_SRC_TMP:.cpp=.o))

TARGET = $(LIB_T2D)/libTomo.so

all : $(TARGET)

# Library
$(TARGET) : $(OBJ) $(OBJ_CU) $(SPALGO_OBJ)
#	$(CXX) -shared $(OBJ) $(OBJ_CU) $(SPALGO_OBJ) -lm -lfftw3 -L $(LIB_CUDA) -lcudart -lgsl -lgslcblas -o $(TARGET)
	$(CXX) $(LINKFLAGS) $(OBJ) $(OBJ_CU) $(SPALGO_OBJ) -lm -lfftw3 -L $(LIB_CUDA) -lcudart -lgsl -lgslcblas -o $(TARGET)

# C++ files
$(OBJ) : $(OBJ_T2D)/%.o : %.cpp $(INC_T2D)/%.hpp
	$(CXX) $(CFLAGS) -I$(INC_EIGEN) -I$(INC_CUDA) -I$(INC_T2D) -I$(KER_T2D) -c $< -o $@

# GPU kernels
$(OBJ_CU) : $(OBJ_T2D)/%.GPU.o : %.GPU.cu $(KER_T2D)/%.Kernel.cu $(KER_T2D)/GPUDevFunc.cu $(KER_T2D)/BlobFunc.cu
	$(NVCC) $(NVCCFLAGS) -I$(KER_T2D) -I$(INC_T2D) -I$(INC_CUDA) -c $< -o $@

# Sparsity related algorithms
$(SPALGO_OBJ) : $(OBJ_T2D)/%.o : SpAlgo/%.cpp $(INC_T2D)/SpAlgo.hpp
	$(CXX) $(CFLAGS) -I$(INC_EIGEN) -I$(INC_CUDA) -I$(INC_T2D) -c $< -o $@

include ../Makefile.phony