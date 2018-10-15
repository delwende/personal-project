#!/bin/sh
#OAR -l walltime=8:00:00
#OAR -O myfiestjob.%jobid%.output
#OAR -E myfiestjob.%jobid%.error


echo
echo OAR_WORKDIR : $OAR_WORKDIR
echo
echo "cat \$OAR_NODE_FILE :"
cat $OAR_NODE_FILE
echo

module load cuda/8.0
module load cudnn/6.0-cuda-8.0
#activate the enviroment eliane(you should have another name) --comment
source ~/eliane/bin/activate
#export LC_ALL=C
#export LANG=fr_FR.latin-1
#export=LC_CTYPE=fr_FR.latin-1

echo "running CRUD DATA SET "
cd /home/ebirba/stage/test1
python3 Codev1.py 
echo "Done"
echo "==================================="