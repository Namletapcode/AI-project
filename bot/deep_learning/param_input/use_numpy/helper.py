import matplotlib.pyplot as plt

plt.ion()

def plot(scores):
    plt.clf()  # Clear current figure
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.legend()
    plt.pause(0.05)  # Pause to allow the plot to update

# testing
if __name__ == '__main__':
    scores = []

    for game in range(100):
        # simulate some training...
        new_score = game % 10 + (game // 10)
        scores.append(new_score)

        plot(scores)
