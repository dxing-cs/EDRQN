# Entropy-regularized Deep Recurrent Q-Network (EDRQN)

This repository contains the code for [*Learning with Generated Teammates to Achieve Type-Free Ad-Hoc Teamwork*](https://www.ijcai.org/proceedings/2021/66) (Dong Xing, Qianhui Liu, Qian Zheng and Gang Pan, IJCAI 2021). Traditional solutions for ad hoc teamwork typically require users to provide a collection of teammate types, which can be hard to guarantee in some scenarios. We provide a general solution to generate diversified teammates automatically to mitigate this problem. 

## Dependencies
- torch=1.7.0 
- numpy=1.18.5 
- tensorboard=2.3.0 (for logging)

These versions are just what I used and not necessarily strict requirements. 

## Run 
```shell
# step 1: edit config file `config_pursuit_edrqn.json`
vim config_pursuit_edrqn.json 

# step 2: train 
python pursuit_train.py 

# step 3: test 
python pursuit_eval.py
```

##  BibTeX
If you find our work helpful in your research, please consider citing our paper:
```shell
@inproceedings{ijcai2021-66,
  title     = {Learning with Generated Teammates to Achieve Type-Free Ad-Hoc Teamwork},
  author    = {Xing, Dong and Liu, Qianhui and Zheng, Qian and Pan, Gang},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {472--478},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/66},
  url       = {https://doi.org/10.24963/ijcai.2021/66},
}
```

If you have any questions, feel free to contact me via `dongxing AT zju.edu.cn`. 