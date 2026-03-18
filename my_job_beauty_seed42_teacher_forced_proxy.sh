#!/bin/bash
#SBATCH -J b42_tf_proxy_v35
#SBATCH -p acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o /data/user/cwu319/RC/poincare/history/output_b42_tf_proxy_v35.txt
#SBATCH -e /data/user/cwu319/RC/poincare/history/error_b42_tf_proxy_v35.txt

set -euo pipefail

cd /data/user/cwu319/RC/poincare
bash scripts/run_beauty_seed42_teacher_forced_proxy.sh
