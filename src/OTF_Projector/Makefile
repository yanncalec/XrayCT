include ../../Makefile.common

#CFLAGS = -Wall -std=gnu99 -g -fPIC #-g Debug information
CFLAGS = -std=gnu99 -fPIC #-g #Debug information

LIBS = -lm # -lml -lgsl -lgslcblas -lcsparse -lcxsparse

SRC = T2DSiddon_OTF.c T2DRegularSampling_OTF.c T2DPixelDriven_OTF.c	\
T2DBlobDriven_OTF.c T2DProjUtil.c
OBJ = $(addprefix $(OBJ_T2D)/, $(SRC:.c=.o))
TARGET = $(LIB_T2D)/libProjOTF.so

$(TARGET) : $(OBJ)
	$(CC) -shared $(LIBS) $(OBJ) -o $@
$(OBJ) : $(OBJ_T2D)/%.o : %.c
	$(CC) $(CFLAGS) -I$(INC_T2D) -c $< -o $@

include ../../Makefile.phony
