#!/bin/bash

cd base
for var3 in c
do 
      sbatch ./base.sbatch
done

# cd ../ranmini_400
# for var1 in a
# do
#     sbatch ./ranmini_400.sbatch
# done

cd ../ranbase
for var2 in b
do
    sbatch ./ranbase.sbatch
done
