#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <time.h>

void dstev_(char* jobz, int* n, double* d, double* e, double* dz,
    int* ldz, double* work, int* info);

#define MASTER_CORE 0
#define MAXITER 100

typedef uint64_t index_type;
typedef uint64_t state_type;

#if !defined(MPI1)
        MPI_Win win1, win2;
        MPI_Aint sizeofdouble;
#endif

struct ed_s {
	double j, gamma;           /* hamiltonian parameters */
	
	unsigned n;                /* total system size in bits count */
	unsigned m;                /* 2^m is the number of cores */
	unsigned nm;               /* local system size in bits count, nm = n - m */
	
	unsigned max_iter;         /* number of iteration to perform */
	unsigned rank;
	unsigned nprocs;
	
	state_type nlstates;       /* local system size, 2^nm */
	
	unsigned *to_nbs;          /* communication pattern */
	unsigned *from_nbs;        /* communication pattern */
	
	double *v0, *v1, *v2;      /* vectors */
	double *vv1, *vv2;         /* buffers */
	
	double *alpha;             /* T matrix entries */
	double *beta;              /* T matrix entries */

        double *evals;

        double *d, *e;             /* matrix elements for the dstev routine */
};

struct ed_s *init(unsigned m, unsigned n, unsigned max_iter, unsigned rank, unsigned nprocs)
{
	unsigned k;
	index_type i;
		
	struct ed_s *ed = (struct ed_s *) malloc(sizeof(struct ed_s));
	assert(ed);
	
	ed->j = 1.0;
	ed->gamma = 1.0;
	
	ed->m = m;
	ed->n = n;
	ed->nm = n - m;
	
	ed->max_iter = max_iter;
	ed->rank = rank;
	ed->nprocs = nprocs;
	
	ed->nlstates = 1 << ed->nm;

	ed->to_nbs = (unsigned *) malloc(m * sizeof(unsigned));
	ed->from_nbs = (unsigned *) malloc(m * sizeof(unsigned));
		
	for (k = 0; k < m; ++k) {
		ed->to_nbs[k] = rank ^ (1 << k);
		ed->from_nbs[k] = ed->to_nbs[k];
	}
	
	/* resize and initialize vectors */
	ed->v0 = (double *) malloc(ed->nlstates * sizeof(double));
	assert(ed->v0);
	for (i = 0; i < ed->nlstates; ++i)
		ed->v0[i] = 0.0;
	ed->v1 = (double *) malloc(ed->nlstates * sizeof(double));
	for (i = 0; i < ed->nlstates; ++i)
		ed->v1[i] = 1.0;
	assert(ed->v1);
	ed->v2 = (double *) malloc(ed->nlstates * sizeof(double));
	assert(ed->v2);
	
	/* resize buffers */
	ed->vv1 = (double *) malloc(ed->nlstates * sizeof(double));
	assert(ed->vv1);
	ed->vv2 = (double *) malloc(ed->nlstates * sizeof(double));
	assert(ed->vv2);

#if !defined(MPI1)
        MPI_Type_size(MPI_DOUBLE, &sizeofdouble);
        MPI_Win_create(ed->vv1, ed->nlstates * sizeofdouble, 
            sizeofdouble, MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
        MPI_Win_fence(0, win1);
        MPI_Win_create(ed->vv2, ed->nlstates * sizeofdouble, 
            sizeofdouble, MPI_INFO_NULL,  MPI_COMM_WORLD, &win2);
        MPI_Win_fence(0, win2);
#endif

	ed->alpha = (double *) malloc(max_iter * sizeof(double));
	assert(ed->alpha);
	ed->beta = (double *) malloc(max_iter * sizeof(double));
	assert(ed->beta);

        ed->d = (double *) malloc(max_iter * sizeof(double));
        assert(ed->d);
        ed->e = (double *) malloc(max_iter * sizeof(double));
        assert(ed->e);

        ed->evals = ed->d;
	
	return ed;
}

void fini(struct ed_s *ed)
{
	free(ed->to_nbs);
	free(ed->from_nbs);
	free(ed->v0);
	free(ed->v1);
	free(ed->v2);
	free(ed->vv1);
	free(ed->vv2);
	free(ed->alpha);
	free(ed->beta);
        free(ed->d);
        free(ed->e);
	free(ed);
}

__inline state_type loc_index2state(index_type index, unsigned nm, unsigned rank)
{
	return index + (rank << nm);
}

__inline index_type state2loc_index(state_type state, index_type nlstates)
{
	return state & (nlstates - 1);
}

__inline state_type flip_state(state_type state, unsigned i)
{
	return state ^ (1 << i);
}

__inline int ph_spin(state_type state, unsigned i)
{
	return 2 * ((int) (state & (1 << i)) >> i) - 1;
}

/* diagonal matrix element */
__inline double diag(state_type state, unsigned n, double j)
{
	unsigned k;
	int s1;
	int d = 0;
	int ps = ph_spin(state, 0);

	for (k = 1; k < n; ++k) {
		s1 = ph_spin(state, k);
		d += ps * s1;
		ps = s1;
	}
	d += ps * ph_spin(state, 0);

	return j * d;
}

__inline void swap(double **v1, double **v2)
{
	double *tmp = *v1;
	*v1 = *v2; *v2 = tmp;
}

void calc_eigenvalues(struct ed_s *ed, unsigned iter)
{
        int n, ldz, info;
        char* job = "N";

        n = iter + 1;
        ldz = 1;
        memcpy(ed->d, ed->alpha, sizeof(double) * (iter + 1));
        memcpy(ed->e, ed->beta + 1, sizeof(double) * iter);

        dstev_("N", &n, ed->d, ed->e, NULL, &ldz, NULL, &info);
//      assert(info == 0);
}

void execute(struct ed_s *ed)
{
	double a, b, ib;
	unsigned k, iter, neighb;
        int mode = 1;
	index_type i;
	state_type s, s1;
	MPI_Request req_send1, req_recv1;
	MPI_Request req_send2, req_recv2;
	MPI_Status stat;
	
	b = 0.0;
	for (i = 0; i < ed->nlstates; ++i)
		b += ed->v1[i] * ed->v1[i];
		
	for (iter = 0; iter < ed->max_iter; ++iter) {
		/* calculate beta */
	  if ( ed->nprocs > 1 ) {
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&b, &ed->beta[iter], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		ed->beta[iter] = sqrt(fabs(ed->beta[iter]));
          } 
	  else {
		ed->beta[iter] = sqrt(b);
          }

		/* normalize v1 */
		ib = 1.0 / ed->beta[iter];
		for (i = 0; i < ed->nlstates; ++i)
			ed->v1[i] *= ib;

		/* send and receive data in advance */
	  if ( ed->nprocs > 1 ) {
#if defined(MPI1)
		MPI_Isend(ed->v1, ed->nlstates, MPI_DOUBLE, ed->to_nbs[0], ed->nm - 1, MPI_COMM_WORLD, &req_send1);
		MPI_Irecv(ed->vv1, ed->nlstates, MPI_DOUBLE, ed->from_nbs[0], ed->nm - 1, MPI_COMM_WORLD, &req_recv1);
#else
                MPI_Put(ed->v1, ed->nlstates, MPI_DOUBLE, ed->to_nbs[0], 0, ed->nlstates, MPI_DOUBLE, win1);
                MPI_Win_fence( 0, win1);
#endif
          }
		/* matrix vector multiplication */
		/* v2 = A * v1, the same core */
		for (i = 0; i < ed->nlstates; ++i) {
			s = loc_index2state(i, ed->nm, ed->rank);

			/* diagonal part */
			ed->v2[i] = diag(s, ed->n, ed->j) * ed->v1[i];

			/* offdiagonal part */
			for (k = 0; k < ed->nm; ++k) {
				s1 = flip_state(s, k);
				ed->v2[i] += ed->gamma * ed->v1[state2loc_index(s1, ed->nlstates)];
			}
		}

		/* matrix vector multiplication */
		/* v2 = A * v1, offdiagonal part, other cores */
		for (k = ed->nm; k < ed->n; ++k) {
			if (k < ed->n - 1) {
				/* send and receive data in advance */
				neighb = k - ed->nm + 1;
				if ( mode ) {
#if defined(MPI1)
				  MPI_Isend(ed->v1, ed->nlstates, MPI_DOUBLE, ed->to_nbs[neighb], k, MPI_COMM_WORLD, &req_send2);
				  MPI_Irecv(ed->vv2, ed->nlstates, MPI_DOUBLE, ed->from_nbs[neighb], k, MPI_COMM_WORLD, &req_recv2);
#else
				  MPI_Put(ed->v1, ed->nlstates, MPI_DOUBLE, ed->to_nbs[neighb], 0, ed->nlstates, MPI_DOUBLE, win2);
				  MPI_Win_fence( 0, win2);
#endif
				} else {
#if defined(MPI1)
				  MPI_Isend(ed->v1, ed->nlstates, MPI_DOUBLE, ed->to_nbs[neighb], k, MPI_COMM_WORLD, &req_send1);
				  MPI_Irecv(ed->vv1, ed->nlstates, MPI_DOUBLE, ed->from_nbs[neighb], k, MPI_COMM_WORLD, &req_recv1);
#else
				  MPI_Put(ed->v1, ed->nlstates, MPI_DOUBLE, ed->to_nbs[neighb], 0, ed->nlstates, MPI_DOUBLE, win1);
				  MPI_Win_fence( 0, win1);
#endif
				}
			}

#if defined(MPI1)
			/* wait until data arrives */
			if ( mode ) {
			  MPI_Wait(&req_recv1, &stat);
			} else {
			  MPI_Wait(&req_recv2, &stat);
			}
#endif

			if ( mode ) {
			  for (i = 0; i < ed->nlstates; ++i) {
			    ed->v2[i] += ed->gamma * ed->vv1[i];
			  }
			  mode = 0;
			} else {
			  for (i = 0; i < ed->nlstates; ++i) {
			    ed->v2[i] += ed->gamma * ed->vv2[i];
			  }
			  mode = 1;
			}

		}

		a = 0.0;
		for (i = 0; i < ed->nlstates; ++i) {
		  a += ed->v1[i] * ed->v2[i];
		}

		/* calculate alpha */
		if ( ed->nprocs > 1 ) {
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&a, &ed->alpha[iter], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		} 
		else {
			ed->alpha[iter] = a;
		}		

		/* v2 = v2 - v0 * beta1 - v1 * alpha1 */
		b = 0.0;
		for (i = 0; i < ed->nlstates; ++i) {
			ed->v2[i] -= ed->v0[i] * ed->beta[iter] + ed->v1[i] * ed->alpha[iter];
			b += ed->v2[i] * ed->v2[i];
		}
		
		/* "shift" vectors */
		swap(&ed->v0, &ed->v1); swap(&ed->v1, &ed->v2);

                if (ed->rank == MASTER_CORE && iter > 0) {
                        calc_eigenvalues(ed, iter);
                        printf("%5i %20.12g\n", iter, ed->evals[0]);
                }

	}

        if (ed->rank == MASTER_CORE)
                printf("Final eigenvalue %20.15g\n", ed->evals[0]);
}

int main(int argc, char *argv[])
{
	int rank, nprocs;
	unsigned m, n, max_iter;
        clock_t t1, t2;
        float ratio, mt_in, mt_out;

	struct ed_s *ed;
	
        ratio = 1./CLOCKS_PER_SEC;

	if (argc < 3) {
	 	puts("ed usage: ed m n\nwhere m=log_2(ncores) and n=log_2(total system size)\n");
		return 1;
	}	

	if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
		fputs("error: initializing mpi", stderr);
		return 1;
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	m = atol(argv[1]);

	if ((1 << m) == nprocs) {
		MPI_Barrier(MPI_COMM_WORLD);
	        t1 = clock();
		
		max_iter = MAXITER;
		n = atol(argv[2]);
		
		ed = init(m, n, max_iter, rank, nprocs);
		assert(ed);
		execute(ed);
		fini(ed);

		MPI_Barrier(MPI_COMM_WORLD);
                t2 = clock();
		
		mt_in = ratio*(long)t2 - ratio*(long)t1;

		MPI_Reduce(&mt_in, &mt_out, 1, MPI_FLOAT, MPI_MAX, MASTER_CORE, MPI_COMM_WORLD);
		
		if (rank == MASTER_CORE)
			printf("time=%g seconds\n", mt_out);
	} else if (rank == MASTER_CORE)
		fprintf(stderr, "error: nprocs=%ui is incompatible with m=%i\n", nprocs, m);
	
#if !defined(MPI1)
        MPI_Win_free( &win1);
        MPI_Win_free( &win2);
#endif

	MPI_Finalize();

	return 0;
}
