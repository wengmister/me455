import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Python script turned out to be a real bad idea. Will probably use a jupyter notebook next time.

resolution = 100

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

def sample_100(true_location):
    """
    This function generates 100 random samples in the range [0,1].
    It uses the sample_by_location function to determine if each sample is a true reading or not.
    """

    # Generate 100 random samples in the range [0,1]
    num_samples = 100
    samples = np.random.rand(num_samples, 2)

    # Get the readings for each sample
    readings = sample_by_location(samples, true_location)
    return samples, readings

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
    samples, readings = sample_100(true_location)

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

def likelihood_from_samples(samples, readings, true_location):
    """
    This function calculates the likelihood of the source location s given the samples and readings.
    It uses calculates the likelihood of the samples given the true location.
    It returns the likelihood value.
    """
    # Calculate the likelihood of the samples given the true location
    # if reading is True, use the location function
    # if reading is False, use 1 - location function
    likelihood = np.prod(location_function(samples[readings], true_location)) * np.prod(1 - location_function(samples[~readings], true_location))
    # Return the likelihood value
    return likelihood

def visualize_likelihood(true_location):
    """
    This function visualizes the likelihood of the source location s given the samples and readings.
    It creates a grid of points and calculates the likelihood at each point using the likelihood_from_samples function.
    It then plots the likelihood as a heatmap.
    """
    likelihood_resolution = 200
    # Generate 100 random samples in the range [0,1]
    samples, readings = sample_100(true_location)
    # Calculate the likelihood of the samples given the true location
    # Create a grid of points
    x = np.linspace(0, 1, likelihood_resolution)
    y = np.linspace(0, 1, likelihood_resolution)
    X, Y = np.meshgrid(x, y)
    # Calculate the likelihood at each point
    Z = np.zeros(X.shape)
    for i in range(likelihood_resolution):
        for j in range(likelihood_resolution):
            location = np.array([X[i, j], Y[i, j]])
            Z[i, j] = likelihood_from_samples(samples, readings, location)
    # Plotting the likelihood
    plt.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='magma')
    plt.colorbar(label='Likelihood')

    # Overlay the sampled points
    colors = ['green' if reading else 'red' for reading in readings]
    plt.scatter(samples[readings, 0], samples[readings, 1], color='green', label='True Readings')
    plt.scatter(samples[~readings, 0], samples[~readings, 1], color='red', label='False Readings')
    plt.scatter(true_location[0], true_location[1], color='blue', marker='x', label='True Location')
    # Add labels and title
    plt.title('Q2.Likelihood of Source Location, resolution = {}'.format(likelihood_resolution))
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(loc='upper right')
    plt.show()

def sample_same_point_100(true_location):
    """
    This function generates 100 measurements on the same random point.
    It uses the sample_by_location function to determine if each sample is a true reading or not.
    """
    # Generate 1 random sample in the range [0,1]
    num_samples = 1
    samples = np.random.rand(num_samples, 2)

    # Repeat the sample 100 times
    samples = np.repeat(samples, 100, axis=0)
    # Get the readings for each sample
    readings = sample_by_location(samples, true_location)
    return samples, readings

def visualize_likelihood_on_same_point_with_subplots(true_location):
    """
    This function visualizes the likelihood of the source location s given the samples and readings
    on the same point, repeated 3 times with different random seeds, and displays them in subplots.
    """
    likelihood_resolution = 100
    seeds = [0, 42, 99]  # Different random seeds
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)  # Create 3 subplots in a single row

    for i, seed in enumerate(seeds):
        np.random.seed(seed)  # Set the random seed
        samples, readings = sample_same_point_100(true_location)  # Generate samples and readings

        # Create a grid of points
        x = np.linspace(0, 1, likelihood_resolution)
        y = np.linspace(0, 1, likelihood_resolution)
        X, Y = np.meshgrid(x, y)

        # Calculate the likelihood at each point
        Z = np.zeros(X.shape)
        for j in range(likelihood_resolution):
            for k in range(likelihood_resolution):
                location = np.array([X[j, k], Y[j, k]])
                Z[j, k] = likelihood_from_samples(samples, readings, location)

        # Plot the likelihood on the current subplot
        ax = axes[i]
        im = ax.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='magma')
        ax.scatter(samples[readings, 0], samples[readings, 1], color='green', label='True Readings')
        ax.scatter(samples[~readings, 0], samples[~readings, 1], color='red', label='False Readings')
        ax.scatter(true_location[0], true_location[1], color='blue', marker='x', label='True Location')

        # Add labels and title for each subplot
        ax.set_title(f'Likelihood (Seed={seed})')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')

    # Add a single color bar outside the subplots
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Likelihood')

    # Add a single legend for all subplots
    handles = [
        mpatches.Patch(color='green', label='True Readings'),
        mpatches.Patch(color='red', label='False Readings'),
        mpatches.Patch(color='blue', label='True Location')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3)

    # Show the plot
    plt.show()

def visualize_sequential_bayes_update(true_location, num_measurements=10):
    """
    Visualizes sequential Bayes updates over `num_measurements` measurements
    from a single fixed sensor location.

    Steps:
      1) Initialize a uniform prior over a grid of (x, y) in [0,1]^2.
      2) Sample a measurement from the sensor at one fixed location.
      3) Update the posterior distribution using Bayes' rule.
      4) Plot the posterior and the measurement in a 2-by-5 grid of subplots.
    """

    # 1) Choose a single sensor location and fix it
    sensor_location = np.array([0.4, 0.5])  # Fixed sensor location
    
    # 2) Setup a grid for source-location hypotheses
    grid_resolution = 50
    x_vals = np.linspace(0, 1, grid_resolution)
    y_vals = np.linspace(0, 1, grid_resolution)
    X, Y = np.meshgrid(x_vals, y_vals)  # shape = (grid_resolution, grid_resolution)
    
    # Flatten for easier likelihood calculation:
    # grid_points is shape (N, 2), where N = grid_resolution^2
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    # 3) Initialize uniform prior over the grid
    prior = np.ones(grid_points.shape[0]) 
    prior /= prior.sum()  # normalize - this will always normalize to 1
    # this is a uniform prior, so it doesn't matter where we start

    # Prepare a 2x5 figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    axes = axes.flatten()  # make it a 1D list of axes

    for i in range(num_measurements):
        # 4) Generate one measurement from the sensor at 'sensor_location'
        #    We'll reuse the sample_by_location function, but it expects arrays:
        reading = sample_by_location(sensor_location.reshape(1, 2), true_location)[0]

        # 5) Compute the likelihood p(measurement | each grid point)
        #    location_function(grid_points, sensor_location) has shape (N,).
        #    But note the signature: location_function(loc, true_location).
        #    We want p(reading|loc = s) = location_function(sensor_location, s).
        #    So we should flip arguments or define a small wrapper:
        sensor_to_grid = location_function(grid_points, sensor_location)
        
        # If reading is True, p(reading|s) = sensor_to_grid
        # If reading is False, p(reading|s) = 1 - sensor_to_grid
        if reading:
            likelihoods = sensor_to_grid
        else:
            likelihoods = 1.0 - sensor_to_grid

        # 6) Bayes' update: posterior ∝ prior * likelihood
        posterior_unnorm = prior * likelihoods
        posterior = posterior_unnorm / posterior_unnorm.sum()

        # 7) Plot the new posterior in the subplot
        ax = axes[i]
        # Reshape posterior back to a 2D grid
        Z = posterior.reshape(grid_resolution, grid_resolution)
        im = ax.imshow(Z, extent=(0,1,0,1), origin='lower', cmap='magma')
        
        # Plot the sensor location and the true location
        # reading -> green (True) or red (False)
        measure_color = 'green' if reading else 'red'
        ax.scatter(sensor_location[0], sensor_location[1], c=measure_color, marker='o', label='Sensor')
        ax.scatter(true_location[0], true_location[1], c='blue', marker='x', label='True Source')
        
        ax.set_title(f"Measurement {i+1}\nReading = {reading}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Update prior for the next iteration
        prior = posterior

    # Add one color bar for all subplots
    cbar = fig.colorbar(im, ax=axes, fraction=0.04)
    cbar.set_label('Posterior Probability')

    # Add an overall title
    fig.suptitle("Q4. Sequential Bayes Update, sensor location = {}".format(sensor_location), fontsize=16)

    # Common legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)
    plt.show()


def visualize_sequential_bayes_sensor_update(true_location, num_measurements=10):
    """
    Visualizes sequential Bayes updates over `num_measurements` measurements
    from a single fixed sensor location.

    Steps:
      1) Initialize a uniform prior over a grid of (x, y) in [0,1]^2.
      2) Sample a measurement from the sensor at one fixed location.
      3) Update the posterior distribution using Bayes' rule.
      4) Plot the posterior and the measurement in a 2-by-5 grid of subplots.
    """

    # 1) Choose a single sensor location and fix it
    sensor_location = np.array([0.4, 0.5])  # Fixed sensor location to start with
    
    # 2) Setup a grid for source-location hypotheses
    grid_resolution = 50
    x_vals = np.linspace(0, 1, grid_resolution)
    y_vals = np.linspace(0, 1, grid_resolution)
    X, Y = np.meshgrid(x_vals, y_vals)  # shape = (grid_resolution, grid_resolution)
    
    # Flatten for easier likelihood calculation:
    # grid_points is shape (N, 2), where N = grid_resolution^2
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    # 3) Initialize uniform prior over the grid
    prior = np.ones(grid_points.shape[0]) 
    prior /= prior.sum()  # normalize - this will always normalize to 1
    # this is a uniform prior, so it doesn't matter where we start

    # Prepare a 2x5 figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    axes = axes.flatten()  # make it a 1D list of axes

    for i in range(num_measurements):
        # 4) Generate one measurement from the sensor at 'sensor_location'
        #    We'll reuse the sample_by_location function, but it expects arrays:
        reading = sample_by_location(sensor_location.reshape(1, 2), true_location)[0]

        # 5) Compute the likelihood p(measurement | each grid point)
        #    location_function(grid_points, sensor_location) has shape (N,).
        #    But note the signature: location_function(loc, true_location).
        #    We want p(reading|loc = s) = location_function(sensor_location, s).
        #    So we should flip arguments or define a small wrapper:
        sensor_to_grid = location_function(grid_points, sensor_location)
        
        # If reading is True, p(reading|s) = sensor_to_grid
        # If reading is False, p(reading|s) = 1 - sensor_to_grid
        if reading:
            likelihoods = sensor_to_grid
        else:
            likelihoods = 1.0 - sensor_to_grid

        # 6) Bayes' update: posterior ∝ prior * likelihood
        posterior_unnorm = prior * likelihoods
        posterior = posterior_unnorm / posterior_unnorm.sum()

        # 7) Plot the new posterior in the subplot
        ax = axes[i]
        # Reshape posterior back to a 2D grid
        Z = posterior.reshape(grid_resolution, grid_resolution)
        im = ax.imshow(Z, extent=(0,1,0,1), origin='lower', cmap='magma')
        
        # Plot the sensor location and the true location
        # reading -> green (True) or red (False)
        measure_color = 'green' if reading else 'red'
        ax.scatter(sensor_location[0], sensor_location[1], c=measure_color, marker='o', label='Sensor')
        ax.scatter(true_location[0], true_location[1], c='blue', marker='x', label='True Source')
        
        ax.set_title(f"Measurement {i+1}\nReading = {reading}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Update prior for the next iteration
        prior = posterior
        # Randomize sensor location by uniform sampling
        sensor_location = np.random.rand(2)


    # Add one color bar for all subplots
    cbar = fig.colorbar(im, ax=axes, fraction=0.04)
    cbar.set_label('Posterior Probability')

    # Add an overall title
    fig.suptitle("Q5. Sequential Bayes Update with sensor resampling", fontsize=16)

    # Common legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)
    plt.show()

def main():
    """
    Main function to run the program.
    It visualizes the background intensity and overlays the sampled points on the same graph.
    """
    # Set the random seed for reproducibility
    np.random.seed(2)
    
    true_location = np.array([0.3, 0.4])

    # Q1: Visualize the background intensity and overlay samples 
    # visualize_background_and_samples(true_location)

    # Q2: Calculate the likelihood of the source location given the samples and readings
    # visualize_likelihood(true_location)

    # Q3: Calculate the likelihood of the source location given the samples and readings on the same point
    # visualize_likelihood_on_same_point_with_subplots(true_location)

    # Q4: Visualize sequential Bayes updates over 10 measurements from a single fixed sensor location
    # visualize_sequential_bayes_update(true_location, num_measurements=10)

    # Q5: Visualize sequential Bayes updates over 10 measurements from a single random sensor location
    visualize_sequential_bayes_sensor_update(true_location, num_measurements=10)

if __name__ == "__main__":
    main()
