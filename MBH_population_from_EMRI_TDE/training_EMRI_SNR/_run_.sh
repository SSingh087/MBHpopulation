#!/bin/sh

rm OUTPUT_FILES/* ERROR_FILES/* LOG_FILES/* workflow_1E5.dag.* *.png

condor_submit_dag workflow_1E5.dag

echo "Ah Finally, it's done. Good Work !!"