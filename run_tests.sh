#!/bin/bash
export TRANSFORMERS_CACHE=/rds/user/ssc42/hpc-work
python3 long_test.py --dola $1
