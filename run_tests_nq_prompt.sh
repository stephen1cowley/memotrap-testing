#!/bin/bash
export TRANSFORMERS_CACHE=/rds/user/ssc42/hpc-work
python3 long_test_nq.py --dola-layers-good $1 --dola-layers-bad $2 --prompt-id $3
