## Overview
This is an article based on the improvement of MBPO. : [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253).
It introduces the standardized flow as a density estimator to estimate the difference between the real and simulated distributions, and uses the difference as a bonus to encourage intelligent agents to explore.

## Dependencies

MuJoCo 1.5 & MuJoCo 2.0

## Usage
> python main_mbpo.py --env_name ${env_name}'-v2' --num_epoch=${num_e} --use_algo 'discriminator'
if you want to quickly reverso context:
> sh run.sh

