#!/bin/bash
#SBATCH -o ./zslurm/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=24mokies2@gmail.com # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jg2619/toolformer-luci/oldtoolvenv/bin/:$PATH
export LD_LIBRARY_PATH=/vol/bitbucket/jg2619/augmenting_llms/dependencies/OpenBlas/lib/:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache
export TORCH_USE_CUDA_DSA=1
/usr/bin/nvidia-smi
echo $(date)
SECONDS=0
source activate
export PYTHONPATH=/vol/bitbucket/jg2619/augmenting_llms/:$PYTHONPATH
echo "Starting 48Gb job"
python test_filtering.py
/usr/bin/nvidia-smi
uptime
duration=$SECONDS
echo $(date)
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


##########3 ssh cloud-vm-45-11.doc.ic.ac.uk


## QUEUED:
# - 1112- WikiSearch trick
# - 1113- Calculator trick
# - 1114- WikiSearch standard

#"http://wordtruthlife.blogspot.com/2018/01/","But holy actually means: separated FROM the world and separated UNTO God!","But holy actually means: separated [WikiSearch(aquinas definition sanctification)] FROM the world and separated UNTO God!","But holy actually means: separated [WikiSearch(aquinas definition sanctification)→ Sanctification . Influenced by the Holiness movement some Pentecostal churches, such as the Church of God in Christ and the Apostolic Faith Church, believe that sanctification is a definitive act of God’s grace and spiritual experience whereby we are made holy subsequent to salvation and prior to the baptism of the Holy Spirit. Reformed Churches] FROM the world and separated UNTO God!","","1.0283279418945312","","“Aquinas definition of sanctification “","aquinas definition sanctification","Truth: January 2018","2019-02-17 01:25:44","sha1:Q2HQ6CWNCZ5U7TKLJGPY23J33XLMTV55","36844","123","wordtruthlife.blogspot.com","crawl-data/CC-MAIN-2019-09/segments/1550247481428.19/wet/CC-MAIN-20190217010854-20190217032854-00602.warc.wet.gz","453","52153","en","0.97","329.5","head"
