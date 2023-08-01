# The official implementation of 
## Sample-Driven Control for Federated Learning for IoT Network with Real-Time Sensing 
Ngoc-Minh Luu, Minh-Duong Nguyen, Ebrahim Bedeer Mohamed, Nguyen Van Duc, and Quoc-Viet~Pham

## Running code
python main.py --memory-size 100000 --initial-steps 50000 --batch-size 32 --max-episode 4000 --max-step 200 --max-episode-eval 5 --max-step-eval 50 --pen-coeff 0.01 --noise 0.01 --poweru-max 10 --f-u-max 2e9  --plot-interval 80000 --user-num 10 --drl-algo ddpg-ei --ai-network cnn --lr-actor 1e-3 --lr-critic 3e-3 --algo SAC

 python main.py --memory-size 100000 --initial-steps 50000 --batch-size 32 --max-episode 1000 --max-step 200 --max-episode-eval 5 --max-step-eval 50 --pen-coeff 0.0100 --poweru-max 10 --f-u-max 2e9 --plot-interval 200000 --user-num 10 --algo DDPG --sample-delay 0.01 --skip-max 100 --L 200 --local-acc 0.325 --global-acc 0.325 --lr-actor 3e-4 --lr-critic 1e-3 --data-size 2800000 --global-acc 0.1
