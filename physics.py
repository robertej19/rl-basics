import numpy as np
import matplotlib.pyplot as plt

#make the font size for plotting very large
plt.rcParams.update({'font.size': 22})

#write variables for gravity, drag coefficent, and launch angle
gravity = 9.8
drag_coefficient = 0.5
launch_angle = 45
initial_velocity = 100

#write a function that calculates the distance a projectile will travel given the variables above
def projectile_distance(velocity, launch_angle, gravity, drag_coefficient):
    launch_angle = np.radians(launch_angle)
    distance = (velocity**2 * np.sin(2 * launch_angle)) / gravity
    return distance

#write a function that calculates the time it takes for a projectile to hit the ground given the variables above
def time_to_hit_ground(velocity, launch_angle, gravity):
    #convert launch angle to radians
    launch_angle = np.radians(launch_angle)
    time = (2 * velocity * np.sin(launch_angle)) / gravity
    return time

# plot the trajectory of the projectile
def plot_trajectory(velocity, launch_angle, gravity, drag_coefficient):
    time = time_to_hit_ground(velocity, launch_angle, gravity)
    time = np.linspace(0, time, 100)
    x = velocity * np.cos(launch_angle) * time
    y = velocity * np.sin(launch_angle) * time - 0.5 * gravity * time**2
    # make the figure size large
    plt.figure(figsize=(16, 12))
    plt.plot(x, y)
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Projectile Trajectory')
    plt.show()

if __name__ == '__main__':
    #distance = projectile_distance(initial_velocity, launch_angle, gravity, drag_coefficient)
    #print('The projectile will travel a distance of', distance, 'meters.')
    #plot_trajectory(initial_velocity, launch_angle, gravity, drag_coefficient)
    
    # make a plot of the distance as a function of launch angle
    launch_angles = np.linspace(0,90, 100)
    distances = [projectile_distance(initial_velocity, angle, gravity, drag_coefficient) for angle in launch_angles]
    plt.figure(figsize=(16, 12))
    plt.plot(launch_angles, distances)
    plt.xlabel('Launch Angle (degrees)')
    plt.ylabel('Distance (m)')
    plt.title('Projectile Distance vs Launch Angle')
    plt.show()