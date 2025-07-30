request.sh:
#!/bin/bash
USER=mhashe4
salloc -t 2:00:00 --gpus=1 --nodes 1 --mem=32G --partition=l4-8-gm192-c192-m768 bash -c "squeue -u $USER; bash scripts/init.sh; exit"

conda init
conda activate fm
# a10g-1-gm24-c32-m128
# a100-8-gm640-c96-m1152
# a10g-4-gm96-c48-m192