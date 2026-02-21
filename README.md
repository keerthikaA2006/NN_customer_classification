# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## Problem Statement
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="1209" height="799" alt="image" src="https://github.com/user-attachments/assets/2a210679-0a23-4590-9a97-84b5380f5907" />


## DESIGN STEPS
### STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

### STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

### STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

### STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.

## PROGRAM
### Name: Keerthika A
### Register Number: 212224220048

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_model(model, train_loader, criterion, optimizer, epochs=100)

```

```python
#function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs=model(inputs)
          loss=criterion(outputs, labels)
          loss.backward()
          optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```
## Dataset Information
<img width="1270" height="260" alt="Screenshot 2026-02-21 190350" src="https://github.com/user-attachments/assets/93b90603-83f6-4265-bb81-1bc9d678d6b4" />


## OUTPUT
### Confusion Matrix
<img width="1036" height="627" alt="Screenshot 2026-02-21 190516" src="https://github.com/user-attachments/assets/3bf7748a-ab92-40ee-8329-53d31a90bc85" />



### Classification Report
<img width="1042" height="604" alt="Screenshot 2026-02-21 190443" src="https://github.com/user-attachments/assets/4e362d11-a7bd-429e-9b86-3b634f066010" />


### New Sample Data Prediction
<img width="924" height="305" alt="Screenshot 2026-02-21 190536" src="https://github.com/user-attachments/assets/ba162c54-38b2-412f-b9e3-3afe08eb0fa1" />


## RESULT
Thus neural network classification model is developded for the given dataset. 
