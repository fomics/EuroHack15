#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
// #include <omp.h>
#include <sys/time.h>

#ifdef _CRAYPAT_CSCS
#include <pat_api.h>
#endif
/*
 * pat_help API regions C .  ; more /opt/cray/perftools/default/include/pat_api.h
 * rm -f *.xf *.ap2 && cc -hnoomp -D_CSCS_ITMAX=1000 -D_CRAYPAT_CSCS jacobi_openmp.c -o l1l2
 * pat_build -f -u l1l2 && sbatch.sh santis 1 l1l2+pat 1 1 1
 * pat_report -T -s traced_functions=show l1l2+pat*.xf >  xf
*/

void init_host();
void finalize_host();
void start_timer();
void stop_timer();
double mytimer_();
void jacobi();

int n, m;
int rank=0;
int size=1;
int iter = 0;
#define CSCS_ITMAX _CSCS_ITMAX
int iter_max = CSCS_ITMAX;
// int iter_max = 1000;

double starttime;
double runtime;

const float pi = 3.1415926535897932384626f;
const float tol = 1.0e-5f;
float residue = 1.0f;

float* A;
float* Anew;
float* y00;

/********************************/
/****         MAIN            ***/
/********************************/
int main(int argc, char** argv)
{
#ifdef _CRAYPAT_CSCS
        int istat=PAT_API_FAIL;
        printf("pat_rec fail=%d ok=%d\n",PAT_API_FAIL,PAT_API_OK);
        istat=PAT_record(PAT_STATE_OFF); //printf("%d: pat_rec=%d\n", __LINE__, istat);
#endif
        n=1024;
        m=1024;
        printf("Jacobi relaxation Calculation: %d x %d mesh ", n, m);

        init_host();

        start_timer();
        jacobi();       // Main calculation
        stop_timer();

        finalize_host();

}

/********************************/
/****        JACOBI           ***/
/********************************/
void jacobi()
{

#ifdef _CRAYPAT_CSCS
        int istat;
        istat=PAT_record(PAT_STATE_ON); //printf("%d: pat_rec=%d\n", __LINE__, istat);
#endif

  while ( residue > tol && iter < iter_max )
    {
      residue = 0.0f;

#ifdef _CRAYPAT_CSCS
      istat=PAT_region_begin( 1, "loop1" ); //printf("%d: pat_rec=%d\n", __LINE__, istat);
#endif

#pragma omp parallel
      {
        float my_residue = 0.f;
        int i,j;
#pragma omp for nowait
        for( j = 1; j < n-1; j++)
          {
            for( i = 1; i < m-1; i++ )
              {
                Anew[j *m+ i] = 0.25f * ( A[j     *m+ (i+1)] + A[j     *m+ (i-1)]
                                          +    A[(j-1) *m+ i]     + A[(j+1) *m+ i]);
                my_residue = fmaxf( my_residue, fabsf(Anew[j *m+ i]-A[j *m + i]));
              }
          }
        

#pragma omp critical
        {
          residue = fmaxf( my_residue, residue);
        }
      }
#ifdef _CRAYPAT_CSCS
      istat=PAT_region_end( 1 ); //printf("%d: pat_rec=%d\n", __LINE__, istat);
      istat=PAT_region_begin( 2, "loop2" ); //printf("%d: pat_rec=%d\n", __LINE__, istat);
#endif
      int i,j;
//#pragma omp parallel for
      for( j = 0; j < n-1; j++)
        {
          for( i = 1; i < m-1; i++ )
            {
              A[j *m+ i] = Anew[j *m+ i];
            }
        }
#ifdef _CRAYPAT_CSCS
        istat=PAT_region_end( 2 ); //printf("%d: pat_rec=%d\n", __LINE__, istat);
        istat=PAT_record(PAT_STATE_OFF); //printf("%d: pat_rec=%d\n", __LINE__, istat);
#endif

      if(rank == 0 && iter % 100 == 0)
        printf("%5d, %0.6f\n", iter, residue);

      iter++;
    }

#ifdef _CRAYPAT_CSCS
        istat=PAT_record(PAT_STATE_OFF); //printf("%d: pat_rec=%d\n", __LINE__, istat);
#endif

}


/********************************/
/**** Initialization routines ***/
/********************************/

void init_host()
{
  A	= (float*) malloc( n*m * sizeof(float) );
  Anew	= (float*) malloc( n*m * sizeof(float) );
  y00	= (float*) malloc( n   * sizeof(float) );

  int i,j;
#ifdef OMP_MEMLOCALITY
#pragma omp parallel for shared(A,Anew,m,n)
  for( j = 0; j < n; j++)
    {
      for( i = 0; i < m; i++ )
        {
          Anew[j *m+ i] 	= 0.0f;
          A[j *m+ i] 		= 0.0f;
        }
    }
#endif //OMP_MEMLOCALITY

  memset(A, 0, n * m * sizeof(float));
  memset(Anew, 0, n * m * sizeof(float));

  // set boundary conditions
#pragma omp parallel for
  for (i = 0; i < m; i++)
    {
            A[0	    *m+ i] = 0.f;
            A[(n-1) *m+ i] = 0.f;
    }

  int j_offset = 0;
  if ( size == 2 && rank == 1 )
    {
      j_offset = n-2;
    }
  for (j = 0; j < n; j++)
    {
      y00[j] = sinf(pi * (j_offset + j) / (n-1));
      A[j *m+ 0] = y00[j];
      A[j *m+ (m-1)] = y00[j]*expf(-pi);
    }

#pragma omp parallel for
  for (i = 1; i < m; i++)
    {
      if (rank == 0)
        Anew[0     *m+ i] = 0.f;
      if (rank == 1 || size == 1)
        Anew[(n-1) *m+ i] = 0.f;
    }
#pragma omp parallel for
  for (j = 1; j < n; j++)
    {
      Anew[j *m+ 0] = y00[j];
      Anew[j *m+ (m-1)] = y00[j]*expf(-pi);
    }
}

void finalize_host()
{
  free(y00);
  free(Anew);
  free(A);
}

/********************************/
/****    Timing functions     ***/
/********************************/
void start_timer()
{
#ifdef USE_MPI
  starttime = MPI_Wtime();
#else
  //starttime = omp_get_wtime();
  starttime = mytimer_();
#endif //USE_MPI
}

// #include <sys/time.h>
double mytimer_()
{
        struct timeval tp;
        struct timezone tzp;
        int i;
        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );    // seconds
}



void stop_timer()
{
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  runtime = MPI_Wtime() - starttime;
#else
  //runtime = omp_get_wtime() - starttime;
  runtime = mytimer_() - starttime;
#endif //USE_MPI

  if (rank == 0) printf(" total: %f s\n", runtime);
}

