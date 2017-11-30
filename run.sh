# model-1: 2scnn_2bilstm
# python main.py --cuda --epoch 1000 --lr 0.005 --log-interval 1 --id 2scnn_2bilstm --batch-size 32 --model-path 2scnn_2bilstm_best_model.pt --eval --test-len 100

# model-2
# python main.py --cuda --epoch 1000 --lr 0.01 --log-interval 1 --id 1lstm_2cnn_100 --batch-size 16 --model-path 1lstm_2cnn_100_best_model.pt

# model-3
# python main.py --cuda --epoch 1000 --lr 0.01 --log-interval 1 --id 1bilstm_2cnn_100 --batch-size 16 --model-path 1bilstm_2cnn_100_best_model.pt

# model-4: 2lcnn_2bilstm_100
# python main.py --cuda --epoch 1000 --lr 0.01 --log-interval 1 --id 2bilstm_2cnn_100 --batch-size 16 --model-path 2bilstm_2cnn_100_best_model.pt

# model-6
python main.py --cuda --lr 0.01 --id 2scnn_2bilstm_scaled_100_random_new --batch-size 32 --epoch 1000 --log-interval 1 --model-path 2scnn_2bilstm_scaled_100_random_new_best_model.pt

# model-7
# python main.py --cuda --epoch 1000 --lr 0.01 --log-interval 1 --id inception_2bilstm_100 --batch-size 16 --train-len 20 --test-len 20