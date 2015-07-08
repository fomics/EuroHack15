! https://bitbucket.org/jgphpc/pug/issues/46/cray_acc_debug-api
program jacobi_acc_kernels_datacopy
use openacc
implicit real(4) (A-H,O-Z)
        integer, parameter :: NN = 1024
        integer, parameter :: NM = 1024

        real(4) A(NN,NM), Anew(NN,NM)
#ifdef _CRAY_ACC_DEBUG
! /opt/cray/cce/default/craylibs/x86-64/include/openacc.h
        integer :: cray_acc_debug_orig
        cray_acc_debug_orig = cray_acc_get_debug_global_level()
        call cray_acc_set_debug_global_level(0)
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
!$acc data copy(A), create(Anew)
do while ( (error > tol) .and. (iter < iter_max) )
  error = 0.0
!$acc kernels
  do j = 2, NM-1
    do i = 2, NN-1
      Anew(i,j) = 0.25 * ( A(i+1,j) + A(i-1,j) + &
                             A(i,j-1) + A(i,j+1) )
      error = max( error, abs(Anew(i,j) - A(i,j)) )
    end do
  end do
!$acc end kernels

#ifdef _CRAY_ACC_DEBUG
call cray_acc_set_debug_global_level(cray_acc_debug_orig)
#endif
!$acc kernels
  do j = 2, NM-1
    do i = 2, NN-1
      A(i,j) = Anew(i,j)
    end do
  end do
!$acc end kernels
#ifdef _CRAY_ACC_DEBUG
call cray_acc_set_debug_global_level(0)
#endif

  if(mod(iter,100) == 0) print 101,iter,error
  iter = iter + 1
end do
!$acc end data

call cpu_time(t2)
print 102,t2-t1

100 format("Jacobi relaxation Calculation: ",i4," x ",i4," mesh")
101 format(2x,i4,2x,f9.6)
102 format("total: ",f9.6," s")
end program jacobi_acc_kernels_datacopy
