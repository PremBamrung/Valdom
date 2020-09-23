#include "trace.h"
#include "common.h"
#include <omp.h>


/* This routine performs the LU factorization of a square matrix by
   block-columns */

void lu_par_loop(Matrix A, info_type info){

  int i, j;

  /* Initialize the tracing system */
  trace_init();
  // i est identique mais chaque thread possède son i
  // créer la pool de thread en dehors de la boucle for permet de 
  //réduire le temps de destruction/créations de threads
  #pragma omp parallel firstprivate(i) 
  {
    for(i=0; i<info.NB; i++){
      
      /* Do the Panel operation on column i */
      // un seul des threads va exécuter l'instruction
      #pragma omp single
      {
        panel(A[i], i, info);
      }
      
      /* Parallelize this loop     */
      // exécution parallèle entre les threads
      #pragma omp  for 
      for(j=i+1; j<info.NB; j++){
        /* Update column j with respect to the result of panel(A, i) */
        update(A[i], A[j], i, j, info);
      }
    }
  }
  /* This routine applies permutations resulting from numerical
     pivoting. It has to be executed sequentially. */
  backperm(A, info);
  
  /* Write the trace in file (ignore) */
  trace_dump("trace_par_loop.svg");
  
  return;
  
}
