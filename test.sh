# Create and activate new virtual environment
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Generate dataset
python mkSeqMNIST.py --N 100 --M 1000 

# Load pre-trained model and test
python main.py --cuda --model-path 2scnn_2bilstm_scaled_100_best_model.pt --eval --test-len 100