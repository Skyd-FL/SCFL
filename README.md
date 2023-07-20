# The official implementation of 
## Sample-Driven Control for Federated Learning for IoT Network with Real-Time Sensing 
Ngoc-Minh Luu, Minh-Duong Nguyen, Ebrahim Bedeer Mohamed, Nguyen Van Duc, and Quoc-Viet~Pham

## Running code
python main.py --memory-size 100000 --initial-steps 50000 --batch-size 32 --max-episode 4000 --max-step 200 --max-episode-eval 5 --max-step-eval 50 --pen-coeff 0.01 --noise 0.01 --poweru-max 10 --f-u-max 2e9  --plot-interval 80000 --user-num 10 --drl-algo ddpg-ei --ai-network cnn --lr-actor 1e-3 --lr-critic 3e-3 --algo SAC

