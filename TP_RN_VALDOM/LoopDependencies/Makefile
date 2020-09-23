CC       = gcc
LINK     = $(CC)

CFLAGS   = -O3 -fopenmp 
LDFLAGS  = $(CFLAGS) 

OBJS = loopdependencies.o

main: $(OBJS)
	$(LINK) -o $@  $(LDFLAGS) $(OBJS) $(LIBS)






clean:
	(rm -f *.o main)



%.o: %.c
	$(CC)  $(CFLAGS)  $(CINCLUDES) $(INCLUDES) $(CDEFS) $(PREC) -c $<






