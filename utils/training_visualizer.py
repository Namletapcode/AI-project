import matplotlib.pyplot as plt

plt.ion()

def plot_training_progress(scores, title='Training...'):
    """
    Plot training progress in real-time
    
    Args:
        scores: List of scores to plot
        title: Title of the plot
    """
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.ylim(ymin=0)
    if scores:  # Add current score if list not empty
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.legend()
    plt.pause(0.05)

# For testing visualization
if __name__ == '__main__':
    test_scores = []
    for game in range(100):
        score = game % 10 + (game // 10)
        test_scores.append(score)
        plot_training_progress(test_scores)