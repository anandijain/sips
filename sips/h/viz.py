"""
visualizing data and helper fxns 

"""
import matplotlib.pyplot as plt


def create_time_steps(length):
    """

    """
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps


def show_plot(plot_data, delta, title):
    """

    """
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = create_time_steps(plot_data[0].shape[0])

    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    return plt


def plot_train_history(history, title):
    """
    simple tf loss plotter

    """
    loss = history.history["loss"]

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, "b", label="Training loss")
    plt.title(title)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    pass
