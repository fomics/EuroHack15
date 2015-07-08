// --- CSCS (Swiss National Supercomputing Center) ---
// https://bitbucket.org/jgphpc/pug/issues/46/cray_acc_debug-api

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#ifdef _CRAY_ACC_DEBUG
#include <openacc.h>
#endif

int main(int argc, char **argv){
    int rank = 0, size = 1;
    int i;
    double a=1.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef _CRAY_ACC_DEBUG
    int cray_acc_debug_orig=0;
    int cray_acc_debug_zero=0;
    cray_acc_debug_orig = cray_acc_get_debug_global_level();
    //ok cray_acc_set_debug_global_level(0);
    cray_acc_set_debug_global_level(cray_acc_debug_zero);
#endif

    //const int N = 4600;
    int N = atoi(argv[1]);  //argv[0] is the program name
    if(!rank) printf("using MPI with %d PEs, N=%d\n", size, N);
#ifdef _OPENACC
    printf("_OPENACC version: %d\n", _OPENACC);    
#endif

    double b[N];
    double c[N]; // = reinterpret_cast<double*>(malloc(N*sizeof(double)));
#pragma acc parallel loop copy(b[0:N-1], c[0:N-1])
    for(i=0; i<N; i++){
        b[i] = i*1.0;
        c[i] = i*100.0;
    }

#ifdef _CRAY_ACC_DEBUG
    cray_acc_set_debug_global_level(cray_acc_debug_orig);
#endif

#pragma acc parallel loop copyin(a) pcopyin(b[0:N-1]) pcopyout(c[0:N-1])
    for( i = 0; i < N; ++i ) {
        c[i] += a*b[i];
}

#ifdef _CRAY_ACC_DEBUG
    //ok cray_acc_set_debug_global_level(0);
    //cray_acc_set_debug_global_level(cray_acc_debug_orig);
    cray_acc_set_debug_global_level(cray_acc_debug_zero);
#endif

    printf("c[0]=%f\n", c[0]);
    printf("c[N-1]=%f\n", c[N-1]);

    // finalize MPI
    MPI_Finalize();

   return 0;
}

