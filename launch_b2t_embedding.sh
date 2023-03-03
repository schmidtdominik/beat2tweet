#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk

#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt=3:00:00
#   no $ -l gpu=true,gpu_type=!(gtx1080ti|titanx)

# ,gpu_type=!(gtx1080ti|rtx2080ti|titanxp|titanx)

#$ -S /bin/bash
#$ -wd /cluster/project2/ifo/
#$ -j y # merge stdout and stderr
#$ -N b2t_embedding

#$ -t 1-100

cd /cluster/project2/ifo/lfo/b2t/beat2tweet
echo "pwd:"
pwd
echo "Task ID: ${SGE_TASK_ID}"

/cluster/project2/ifo/anaconda3/bin/conda run -n b2t python mtg-jamendo-feature-extraction.py ${SGE_TASK_ID}



# #  ${SGE_TASK_ID}
