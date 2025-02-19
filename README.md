# Water Level Prediction using CNN-LSTM-KAN

This project applies a deep learning model, CNN-LSTM-KAN, to predict water levels in Venice. It combines convolutional neural networks (CNN), long short-term memory (LSTM) networks, and Kolmogorov-Arnold Networks (KAN) to enhance time series prediction accuracy.

## 1. Project Overview

The main goal is to predict water levels using historical tide data and meteorological features. The dataset consists of multiple features, including past water levels, tide, wind speed, atmospheric pressure, temperature, rainfall, radiation, and humidity.

## 2. Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn plotly
```

Required libraries:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib.pyplot`: Visualization
- `seaborn`: Enhanced statistical visualizations
- `torch`: PyTorch for deep learning
- `sklearn.preprocessing`: Data normalization
- `sklearn.metrics`: Model evaluation
- `math`: Mathematical functions
- `time`: Time tracking
- `plotly.graph_objects`: Interactive visualization

## 3. Data Preprocessing

1. Load the dataset (`2years.csv`).
2. Convert date format for time series processing.
3. Visualize the original water level trend.
4. Normalize feature values between [-1,1].
5. Create sliding window sequences for training and testing.

## 4. Model Architecture

The CNN-LSTM-KAN model consists of:
- **CNN Layer**: Extracts local patterns from time-series data.
- **LSTM Layer**: Captures long-term dependencies.
- **KAN Layer**: Enhances non-linear feature learning.

### Model Definition:
```python
class CNN_LSTM_KAN(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, out_channels, output_size):
        super(CNN_LSTM_KAN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.kan = KAN(width=[hidden_size, output_size])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.kan(x[:, -1, :])
        return x
```

## 5. Model Training

- Data is split into 80% training and 20% testing.
- Mean Squared Error (MSE) loss is used.
- Adam optimizer is applied with learning rate `0.01` and weight decay `1e-4`.
- The model is trained for `100` epochs.

## 6. Evaluation Metrics

- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Mean Squared Error (MSE)
- Nash-Sutcliffe Efficiency (NSE)

## 7. Results Visualization

- Loss curve over epochs
- Predicted vs. actual water levels using `plotly`

## 8. Usage

### Run the script:
```bash
python main.py
```

### Modify Hyperparameters:
Adjust parameters in the model initialization section:
```python
input_dim = scaled_features_df.shape[1]-1
hidden_dim = 64
num_layers = 3
output_dim = 1
num_epochs = 100
```

## 9. Acknowledgments

This project integrates traditional deep learning with KANs to improve hydrological predictions. Special thanks to research contributors in deep learning for time-series forecasting.

## 10. Future Work

- Experiment with alternative architectures.
- Implement attention mechanisms.
- Extend to other hydrological datasets.

