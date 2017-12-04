# Create and activate new virtual environment
echo "=> Setting up environment ..."
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
echo "=> Environment set up"

# echo "=> Downloading MNIST dataset and generating sequence test set ..."
# # Generate dataset
# python mkSeqMNIST.py --N 100 --M 1000 
# echo "=> Dataset done"

echo "=> Running model and testing ..."
# Load pre-trained model and test
python main.py --cuda --model-path 2scnn_2bilstm_scaled_100_best_model.pt --eval --test-len 100