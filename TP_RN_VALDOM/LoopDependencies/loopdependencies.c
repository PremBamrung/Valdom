#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "omp.h"


int main(){

  int n, i;
  double *a, *b, *a1, *b1;


  /* Initialize the data */
  n = 1000;
  a  = (double*)malloc(n*sizeof(double));
  b  = (double*)malloc(n*sizeof(double));
  a1 = (double*)malloc(n*sizeof(double));
  b1 = (double*)malloc(n*sizeof(double));
  for(i=0; i<n; i++){
    a[i] = rand();
    b[i] = rand();
  }
  memcpy(a1, a, n*sizeof(double));
  memcpy(b1, b, n*sizeof(double));


  /* Sequential version */
  for(i=0; i<n; i++){
    b[i] = a[i]+1;
    if(i != 0)
      a[i] = b[i]-b[i-1];
  }


  /* Parallel, incorrect version.  
     Modify this code in order to provide a correct parallel
     implementation. Pay attention to the loop-carried
     dependencies. */
  #pragma omp parallel for
  for(i=0; i<n; i++){
    b1[i] = a1[i]+1;
  }

  #pragma omp parallel for
  for (i=0; i<n; i++){
    if(i != 0)
      a1[i] = b1[i]-b1[i-1];
  }


  

  for(i=0; i<n; i++)
    if(a[i] != a1[i]){
      printf("The result is not correct\n");
      return 1;
    }

  printf("The result is correct\n");
  return 0;
}


