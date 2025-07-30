#!/bin/bash
USER=mhashe4
# use squeue to acquire the first jobid
FIRST_JOBID=$(squeue -u $USER -h -o "%i" | tail -n 1)
# check jobid
if [ -z "$FIRST_JOBID" ]; then
    echo "No jobs found for user $USER."
else
    echo "The first jobid for user $USER is: $FIRST_JOBID"
fi
# enter node
srun --jobid $FIRST_JOBID --pty --preserve-env bash