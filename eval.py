
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
# X_test_raw_tensors is List[torch.Tensor], y_test_raw_tensors is List[torch.Tensor]
X_test_raw_tensors, y_test_raw_tensors = create_test_dataset()

# Determine the maximum card length used during training (from create_train_dataset in utils)
# Assuming max_train_card is 10 based on create_train_dataset implementation
max_card_length = 10

# Group samples by their original cardinality and pad/truncate them to max_card_length
grouped_test_data = {}
for i in range(len(X_test_raw_tensors)):
    original_card_length = X_test_raw_tensors[i].size(0)
    current_x_tensor = X_test_raw_tensors[i]
    current_y_tensor = y_test_raw_tensors[i]

    # Pad if shorter, truncate if longer, to match model input expectation (length 10)
    padded_x_tensor = torch.zeros(max_card_length, dtype=torch.long)
    if original_card_length <= max_card_length:
        padded_x_tensor[:original_card_length] = current_x_tensor
    else:
        padded_x_tensor = current_x_tensor[:max_card_length]

    # Store the processed tensor and its corresponding label
    if original_card_length not in grouped_test_data:
        grouped_test_data[original_card_length] = {'X': [], 'y': []}

    grouped_test_data[original_card_length]['X'].append(padded_x_tensor)
    grouped_test_data[original_card_length]['y'].append(current_y_tensor)

# Convert grouped lists of tensors into stacked tensors for efficient batching
# X_test_processed will be a list of 2D tensors, y_test_processed a list of 1D tensors
X_test_processed = []
y_test_processed = []
cards = [] # This list will contain the actual original cardinalities tested
for card_len in sorted(grouped_test_data.keys()):
    X_test_processed.append(torch.stack(grouped_test_data[card_len]['X']))
    y_test_processed.append(torch.stack(grouped_test_data[card_len]['y']))
    cards.append(card_len) # Store the actual cardinality

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

for i in range(len(cards)): # Iterate over different actual cardinalities
    y_pred_deepsets = list()
    y_pred_lstm = list()

    current_X_test_group = X_test_processed[i] # X_test tensors for the current cardinality group
    current_y_test_group = y_test_processed[i] # y_test tensors for the current cardinality group

    # Loop through batches for the current cardinality group
    for j in range(0, current_X_test_group.size(0), batch_size): # Use size(0) for tensor length

        ############## Task 6
        x_batch = current_X_test_group[j:j+batch_size].to(device)
        # Ensure y_batch is float and has an extra dimension for L1Loss consistency
        y_batch = current_y_test_group[j:j+batch_size].unsqueeze(1).to(device)

        with torch.no_grad(): # Disable gradient calculations during inference
            output_deepsets = deepsets(x_batch)
            output_lstm = lstm(x_batch)

        y_pred_deepsets.append(output_deepsets.cpu())
        y_pred_lstm.append(output_lstm.cpu())


    y_pred_deepsets = torch.cat(y_pred_deepsets)
    # Squeeze to remove the last dimension (1) and match y_true_current's shape for metrics
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy().squeeze()
    y_true_current = current_y_test_group.cpu().numpy() # Convert labels to numpy for sklearn metrics

    acc_deepsets = accuracy_score(y_true_current, np.round(y_pred_deepsets))
    mae_deepsets = mean_absolute_error(y_true_current, y_pred_deepsets)
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)

    y_pred_lstm = torch.cat(y_pred_lstm)
    # Squeeze to remove the last dimension (1) and match y_true_current's shape for metrics
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy().squeeze()

    acc_lstm = accuracy_score(y_true_current, np.round(y_pred_lstm))
    mae_lstm = mean_absolute_error(y_true_current, y_pred_lstm)
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
# Plotting results

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot(cards, results['deepsets']['acc'], marker='o', label='DeepSets Accuracy')
plt.plot(cards, results['lstm']['acc'], marker='x', label='LSTM Accuracy')
plt.title('Model Accuracy vs. Cardinality')
plt.xlabel('Cardinality (Number of elements in set)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot MAE
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
plt.plot(cards, results['deepsets']['mae'], marker='o', label='DeepSets MAE')
plt.plot(cards, results['lstm']['mae'], marker='x', label='LSTM MAE')
plt.title('Model MAE vs. Cardinality')
plt.xlabel('Cardinality (Number of elements in set)')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print numerical results for clarity
print("\n--- DeepSets Results ---")
for i, card_val in enumerate(cards):
    print(f"Cardinality {card_val}: Accuracy = {results['deepsets']['acc'][i]:.4f}, MAE = {results['deepsets']['mae'][i]:.4f}")

print("\n--- LSTM Results ---")
for i, card_val in enumerate(cards):
    print(f"Cardinality {card_val}: Accuracy = {results['lstm']['acc'][i]:.4f}, MAE = {results['lstm']['mae'][i]:.4f}")
