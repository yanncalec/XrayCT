# sys :
# 	@echo $(SYSNAME)

include ../Makefile.common

#CFLAGS = -O2
CFLAGS = -fPIC
LINKFLAGS = -shared

SRC = BlobCG.cpp BlobImgDisp.cpp BlobTV.cpp PixBesov.cpp		\
PixImgDisp.cpp SgProc.cpp BlobImgDebias.cpp BlobL1FISTA.cpp		\
Config2Txt_ia.cpp Config2Txt.cpp PixARTUR.cpp PixCG.cpp PixTV.cpp	\
SgSim.cpp BlobTVADM.cpp BlobTVL1.cpp BlobL1PADM.cpp PixL1FISTA.cpp	\
PixL1PADM.cpp ImageQA.cpp


OBJ = $(addprefix $(OBJ_T2D)/, $(SRC:.cpp=.o))
RUNS = $(addprefix $(BIN_T2D)/, $(SRC:.cpp=.run))
LIBCMD = $(LIB_T2D)/libCmdTools.so

all : $(LIBCMD) $(RUNS)

$(LIBCMD) : $(OBJ_T2D)/CmdTools.o 
#	$(CXX) -shared -lm -L$(LIB_CUDA) -lcudart -lX11 -lpthread $< -o $@
	$(CXX) $(LINKFLAGS) -lm -L$(LIB_CUDA) -lcudart -lX11 -lpthread $< -o $@

$(OBJ_T2D)/CmdTools.o : CmdTools.cpp
	$(CXX) $(CFLAGS) -I$(INC_CIMG) -I$(INC_EIGEN) -I$(INC_CUDA) -I$(INC_T2D) -I$(INC_TCLAP) -c $< -o $@

$(OBJ) : $(OBJ_T2D)/%.o : %.cpp
#	$(CXX) $(CFLAGS) -I$(INC_CIMG) -I$(INC_EIGEN) -I$(INC_CUDA) -I$(INC_T2D) -I$(INC_TCLAP) -c $< -o $@
	$(CXX) -O2 -I$(INC_CIMG) -I$(INC_EIGEN) -I$(INC_CUDA) -I$(INC_T2D) -I$(INC_TCLAP) -c $< -o $@

#$(RUNS) : $(OBJ)
$(RUNS) : $(BIN_T2D)/%.run : $(OBJ_T2D)/%.o $(LIB_T2D)/libTomo.so $(LIBCMD)
	$(CXX) -L$(LIB_T2D) -lTomo -lProjOTF -lCmdTools $< -o $@

include ../Makefile.phony
