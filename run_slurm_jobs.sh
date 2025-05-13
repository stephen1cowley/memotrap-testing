#!/bin/bash
source ~/pytorch-env/bin/activate

sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/none_none.wilkes3
sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/none_low.wilkes3
# sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/none_high.wilkes3
sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/low_none.wilkes3
sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/low_low.wilkes3
# sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/low_high.wilkes3
# sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/high_none.wilkes3
# sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/high_low.wilkes3
# sbatch /home/ssc42/ondemand/data/sys/myjobs/projects/default/$1/high_high.wilkes3
