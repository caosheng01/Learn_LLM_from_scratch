import torch
import torch.nn.functional as F

# Define sample tensor
x1 = torch.tensor(1)
x2 = torch.tensor(2)
Y = torch.tensor(2)

# Define weight, bias and learning rate
w11 = torch.tensor(1.0, requires_grad=True)
w12 = torch.tensor(1.0, requires_grad=True)
w21 = torch.tensor(1.0, requires_grad=True)
w22 = torch.tensor(-1.0, requires_grad=True)
wa1 = torch.tensor(1.0, requires_grad=True)
wa2 = torch.tensor(-1.0, requires_grad=True)
b11, b12, b21 = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
learning_rate = 0.01

# Step 1/3: Forward propagation
z1 = w11 * x1 + w12 * x2 + b11
a1 = F.relu(z1)
z2 = w21 * x1 + w22 * x2 + b12
a2 = F.relu(z2)
y = wa1 * a1 + wa2 * a2 + b21

# Calculate loss
loss = ((y - Y)**2)/2

# Step 2/3: Backward propagation
# calculate gradients
w11_grad = (y - Y) * wa1 * 1 * x1
print(w11_grad)

# Step 3/3: Update parameters
w11 = w11 - learning_rate * w11_grad
print(w11)



# # calculate all weight's gradients
# loss.backward()
# print(w11.grad)
# # with torch.no_grad():
# #     w11 -= learning_rate * w11.grad
# #
# # print(w11.grad)

