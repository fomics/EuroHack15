program jacobi1

! https://bitbucket.org/jgphpc/pug/issue/45/perftools-api

!use mpi
implicit real(4) (A-H,O-Z)
#ifdef _CRAYPAT_CSCS
include 'pat_apif.h'
#endif
integer, parameter :: NN = 1024
integer, parameter :: NM = 1024
real(4) A(NN,NM), Anew(NN,NM)

#ifdef _CRAYPAT_CSCS
        integer :: istat
        !call MPI_INIT (istat)
        call PAT_record(PAT_STATE_OFF, istat); 
        !print *,"pat_rec1=",istat,"f=",PAT_API_FAIL,"ok=",PAT_API_OK
#endif

        iter_max = 1000
        tol   = 1.0e-6
        error = 1.0
        A(1,:) = 1.0
        A(2:NN,:) = 0.0
        Anew(1,:) = 1.0
        Anew(2:NN,:) = 0.0
        print 100,NN,NM

        call cpu_time(t1)
        iter = 0
        do while ( (error > tol) .and. (iter < iter_max) )
        error = 0.0

#ifdef _CRAYPAT_CSCS
        call PAT_record(PAT_STATE_ON, istat); !print *,"pat_rec2=",istat
        call PAT_record(PAT_STATE_ON, istat); !print *,"pat_rec2=",istat
        call PAT_region_begin( 1, "loop1", istat ); !print *,"pat_rec3=",istat
#endif
        do j = 2, NM-1
        do i = 2, NN-1
                Anew(i,j) = 0.25 * ( A(i+1,j) + A(i-1,j) + &
                         A(i,j-1) + A(i,j+1) )
                error = max( error, abs(Anew(i,j) - A(i,j)) )
        end do
        end do
#ifdef _CRAYPAT_CSCS
        call PAT_region_end ( 1, istat ); !print *,"pat_rec4=",istat
        call PAT_region_begin ( 2, "loop2", istat ); !print *,"pat_rec5=",istat
#endif    
        do j = 2, NM-1
        do i = 2, NN-1
                A(i,j) = Anew(i,j)
        end do
        end do

#ifdef _CRAYPAT_CSCS
        call PAT_region_end ( 2, istat ); !print *,"pat_rec6=",istat
        call PAT_record(PAT_STATE_OFF, istat); !print *,"pat_rec7=",istat
#endif    

        if(mod(iter,100) == 0) print 101,iter,error
        iter = iter + 1
        end do

        call cpu_time(t2)
        print 102,t2-t1
!#ifdef _CRAYPAT_CSCS
!call MPI_FINALIZE (istat)
!#endif

100 format("Jacobi relaxation Calculation: ",i4," x ",i4," mesh")
101 format(2x,i4,2x,f9.6)
102 format("total: ",f9.6," s")

end program jacobi1
