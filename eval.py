import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_train_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):

        x_batch_np = X_test[i][j:j+batch_size]
        y_batch_np = y_test[i][j:j+batch_size]

        x_batch = torch.tensor(x_batch_np, dtype=torch.long).to(device)
        # y_batch = torch.tensor(y_batch_np, dtype=torch.float32).to(device) # Not needed for prediction

        with torch.no_grad():
            output_deepsets = deepsets(x_batch)
            output_lstm = lstm(x_batch)

        y_pred_deepsets.append(output_deepsets.cpu())
        y_pred_lstm.append(output_lstm.cpu())

    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

    # acc_deepsets = # Accuracy is not suitable for this regression task
    mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)

    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()

    # acc_lstm = # Accuracy is not suitable for this regression task
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
    results['lstm']['mae'].append(mae_lstm)



plt.figure(figsize=(12, 7))
plt.plot(cards, results['deepsets']['mae'], label='DeepSets MAE', marker='o')
plt.plot(cards, results['lstm']['mae'], label='LSTM MAE', marker='o')
plt.xlabel('Number of Cards (Cardinality)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE vs. Cardinality for DeepSets and LSTM')
plt.xticks(cards)
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Summary of Results ---")
print("Cardinality | DeepSets MAE | LSTM MAE")
print("-----------------------------------")
for idx, card in enumerate(cards):
    print(f"{card:<11} | {results['deepsets']['mae'][idx]:<12.4f} | {results['lstm']['mae'][idx]:<9.4f}")