#!/bin/bash

# ONLY ACTIVATE ONE RUNTIME COLLECTION METHOD AT A TIME!!!

# A name for the files
jobstem=$(printf "o_computeprof.%03d" $ALPS_APP_PE)

# NVIDIA COMPUTE_PROFILER: set this 1 to activate
export COMPUTE_PROFILE=1
# Collect output in separate files, one per process
export COMPUTE_PROFILE_LOG=./${jobstem}
# Tune what is collected (optional)
export COMPUTE_PROFILE_CONFIG=compute_profile_config

# Collect CCE runtime information: set this 1,2,3 to activate 
export CRAY_ACC_DEBUG=0

# Collect PGI runtime information: set this 1,3 etc. to activate 
export PGI_ACC_NOTIFY=0

# Now execute binary with appropriate options
#   Pipe STDERR to separate files
# (to catch CRAY_ACC_DEBUG, PGI_ACC_NOTIFY commentary) 
exec $* 2> ${jobstem}

