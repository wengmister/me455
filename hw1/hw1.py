import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

resolution = 1000

def sample_by_location(sample, true_location):
    """
    This function takes a sample point and a true location as input, and calculates the probability if the sensor returns a true reading.
    It generates a random number and compares it with the location function of the sensor. 
    - If the random number is less than the location function, it returns True, indicating a true reading.
    - If the random number is greater than or equal to the location function, it returns False, indicating a false reading.

    input args:
    - sample: 2d np array of (x,y) coordinates representing the sample point.
    - true_location: 2d np array of (x,y) coordinates representing the true location of the sensor.
    """

    f = location_function(sample, true_location)
    random_number = np.random.rand(len(sample))
    return random_number < f

def location_function(loc, true_location):
    """
    This function calculates intensity of a sensor based on its distance from the true location.

    input args:
    - loc: 2d np array of (x,y) coordinates representing the sensor's location.
    - true_location: 2d np array of (x,y) coordinates representing the true location of the sensor.
    """

    return np.exp(-100*(np.linalg.norm(loc - true_location, axis=1) - 0.2)**2)

def plot_100_samples(true_location):
    """
    This function generates 100 random samples in the range [0,1] and plots the results.
    It uses the sample_by_location function to determine if each sample is a true reading or not.
    """

    # Generate 100 random samples in the range [0,1]
    num_samples = 100
    samples = np.random.rand(num_samples, 2)

    # Get the readings for each sample
    readings = sample_by_location(samples, true_location)

    # Plot the results
    plt.scatter(samples[:, 0], samples[:, 1], c=readings, cmap='coolwarm', label='Sampled Points')
    plt.colorbar(label='True Reading')
    plt.title('Sampled Points with True Readings')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.scatter(true_location[0], true_location[1], color='blue', marker='x', label='True Location')
    plt.legend()
    plt.show()

def visualize_background(true_location):
    """
    This function visualizes the background intensity based on the location function.
    It creates a grid of points and calculates the intensity at each point using the location_function."""
    
    # Create a grid of points
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    # Visualize the grid and vary intensity based on location_function
    Z = location_function(np.column_stack((X.ravel(), Y.ravel())), true_location)
    Z = Z.reshape(X.shape)

    # Plotting the intensity based on location function
    plt.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='magma')
    plt.colorbar(label='Intensity')
    plt.title('Intensity based on location function')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.scatter(true_location[0], true_location[1], color='blue', marker='x', label='True Location')
    plt.legend()
    plt.show()


def visualize_background_and_samples(true_location):
    """
    This function visualizes the background intensity based on the location function
    and overlays the sampled points on the same graph.
    """
    # Create a grid of points
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    Z = location_function(np.column_stack((X.ravel(), Y.ravel())), true_location)
    Z = Z.reshape(X.shape)

    # Plot the background intensity
    plt.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='magma')
    plt.colorbar(label='Intensity')

    # Generate 100 random samples in the range [0,1]
    num_samples = 100
    samples = np.random.rand(num_samples, 2)

    # Get the readings for each sample
    readings = sample_by_location(samples, true_location)

    # Map readings to colors: True -> green, False -> red
    colors = ['green' if reading else 'red' for reading in readings]

    # Overlay the sampled points
    plt.scatter(samples[:, 0], samples[:, 1], c=colors, label='Sampled Points')
    plt.scatter(true_location[0], true_location[1], color='blue', marker='x', label='True Location')

    # Add custom legend for positive and negative points
    positive_patch = mpatches.Patch(color='green', label='Positive Reading')
    negative_patch = mpatches.Patch(color='red', label='Negative Reading')
    true_location_patch = mpatches.Patch(color='blue', label='True Location')

    # Add labels and title
    plt.title('Q1.Background Intensity with Sampled Points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(handles=[positive_patch, negative_patch, true_location_patch], loc='upper right')
    plt.show()

def main():
    """
    Main function to run the program.
    It visualizes the background intensity and overlays the sampled points on the same graph.
    """
    # Set the random seed for reproducibility
    np.random.seed(0)
    
    true_location = np.array([0.3, 0.4])

    # Visualize the background intensity and overlay samples
    visualize_background_and_samples(true_location)

if __name__ == "__main__":
    main()


