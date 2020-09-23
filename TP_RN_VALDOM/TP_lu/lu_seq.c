#include "trace.h"
#include "common.h"

/* This is a sequential routine for the LU factorization of a square
   matrix in block-columns */
void lu_seq(Matrix A, info_type info){


  int i, j;

  trace_init();
  #pragma omp parallel
  // un seul thread lit l'instruction
  #pragma omp single 
  for(i=0; i<info.NB; i++){
    /* Do the panel */
    // dépend de l'entré et la sortie de A[i]
    #pragma omp task depend(inout:A[i])
    panel(A[i], i, info);
    
    for(j=i+1; j<info.NB; j++){
      /* Do all the correspondint updates */
      // Dépend de l'entrée A[i] et de la sortie A[j]
      #pragma omp task depend(in:A[i],A[j]) depend(out:A[j])
      update(A[i], A[j], i, j, info);
    }    
  }
  
  /* Do row permutations resulting from the numerical pivoting */
  backperm(A, info);

  trace_dump("trace_seq.svg");

  return;

}

