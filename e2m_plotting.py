    # Access the info logs after training
    info_logs = wrapped_env.get_info_logs()
    # Extract rx, ry, and rz
    rx_out = [log['rx'] for log in info_logs]
    ry_out = [log['ry'] for log in info_logs]
    rz_out = [log['rz'] for log in info_logs]
    rx_out = np.array(rx_out)
    ry_out = np.array(ry_out)
    rz_out = np.array(rz_out)
    
    # Plotting the trajectory in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(rx_out, ry_out, marker='o')

    ax.set_xlabel('rx')
    ax.set_ylabel('ry')
    ax.set_title('Spacecraft Trajectory')

    plot_path = os.path.join("Plots", "test_path1.png")
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")

# TODO: Plot trajectory
