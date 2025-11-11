import numpy as np
import matplotlib.pyplot as plt

weights = np.array([0.5, 0.5])
threshold = 0.5
learning_rate = 0.1

# change as per needed
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

print("Training Perceptron for AND Gate\n")

for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    errors = 0
    
    for i in range(len(inputs)):

        summation = np.dot(inputs[i], weights)
        output = 1 if summation > threshold else 0

        if output != targets[i]:
            error = targets[i] - output
            weights += learning_rate * error * inputs[i]
            errors += 1
            print(f"  Input: {inputs[i]} -> Output: {output}, Expected: {targets[i]} ❌")
        else:
            print(f"  Input: {inputs[i]} -> Output: {output} ✓")
    
    if errors == 0:
        print(f"\nTraining completed at epoch {epoch + 1}!")
        break

print(f"\nFinal weights: {weights}")
print(f"Threshold: {threshold}")

plt.figure(figsize=(8, 6))

for i in range(len(inputs)):
    color = 'blue' if targets[i] == 1 else 'red'
    marker = 'o' if targets[i] == 1 else 'x'
    plt.scatter(inputs[i][0], inputs[i][1], c=color, marker=marker, s=200)

x = np.linspace(-0.5, 1.5, 100)
y = (threshold - weights[0] * x) / weights[1]
plt.plot(x, y, 'g-', linewidth=2, label='Decision Boundary')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron Decision Boundary - AND Gate')
plt.legend(['Decision Boundary', 'Class 1', 'Class 0'])
plt.grid(True)
plt.show()
