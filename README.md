# The official implementation of 
## Energy-Efficient and Real-Time Sensing for Federated Continual Learning via Sample-Driven Control
Ngoc-Minh Luu, Minh-Duong Nguyen, Ebrahim Bedeer Mohamed, Dinh Thai Hoang, Diep-N.-Nguyen, Nguyen Van Duc, and Quoc-Viet~Pham
Transactions on Mobile Computing

https://arxiv.org/abs/2310.07497

Anyone re-use the code please cite the followings: 

```
@article{luu2023sample,
  title={Energy-Efficient and Real-Time Sensing for Federated Continual Learning via Sample-Driven Control},
  author={Luu, Minh Ngoc and Nguyen, Minh-Duong and Bedeer, Ebrahim and Nguyen, Van Duc and Hoang, Dinh Thai and Nguyen, Diep N and Pham, Quoc-Viet},
  journal={arXiv preprint arXiv:2310.07497},
  month={Oct.},
  year={2023}
}
```

## Running code
####
python main.py --memory-size 100000 --initial-steps 50000 --batch-size 32 --max-episode 4000 --max-step 200 --max-episode-eval 5 --max-step-eval 50 --pen-coeff 0.01 --noise 0.01 --poweru-max 10 --f-u-max 2e9  --plot-interval 80000 --user-num 10 --drl-algo ddpg-ei --ai-network cnn --lr-actor 1e-3 --lr-critic 3e-3 --algo SAC

####
 python main.py --memory-size 100000 --initial-steps 50000 --batch-size 32 --max-episode 1000 --max-step 200 --max-episode-eval 5 --max-step-eval 50 --pen-coeff 0.0100 --poweru-max 10 --f-u-max 2e9 --plot-interval 200000 --user-num 10 --algo DDPG --sample-delay 0.01 --skip-max 100 --L 200 --local-acc 0.325 --global-acc 0.325 --lr-actor 3e-4 --lr-critic 1e-3 --data-size 2800000 --global-acc 0.1
