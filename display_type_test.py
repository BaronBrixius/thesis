import matplotlib.pyplot as plt

def logistic_map(x, alpha=1.7):
    """Logistic map equation."""
    return 1 - alpha * x**2

# Parameters
alpha = 1.7  # Logistic map parameter
iterations = 20  # Number of iterations
start_values = [0.09, 0.1, 0.11]  # Starting values

# Create the plot
plt.figure(figsize=(8, 6))

for start in start_values:
    x = start
    trajectory = []
    for _ in range(iterations):
        trajectory.append(x)
        x = logistic_map(x, alpha)
    plt.plot(range(iterations), trajectory, label=f"Start: {start}")

# Customize plot
plt.title("Logistic Map Trajectories")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.grid(True)

# Show the plot
plt.show()
