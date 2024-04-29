import matplotlib.pyplot as plt

# Data for plotting
metrics = ['Label Loss', 'Next Loss', 'Total Loss']
loss_values = [13.676788330078125, 1.1920928244535389e-07, 13.676788449287407]
perplexity_values = [870469.7871126005, 1.0000001192092896, 870469.8908806854]

fig, ax1 = plt.subplots()

# Bar graph for losses
color = 'tab:red'
ax1.set_xlabel('Metric')
ax1.set_ylabel('Loss', color=color)
bars = ax1.bar(metrics, loss_values, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

# Label each bar with its value
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

# Create a second y-axis for perplexity
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Perplexity', color=color)
line, = ax2.plot(metrics, perplexity_values, color=color, marker='o', label='Perplexity')
ax2.tick_params(axis='y', labelcolor=color)

# Label each line point with its value
for i, txt in enumerate(perplexity_values):
    ax2.annotate(round(txt, 2), (metrics[i], perplexity_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding title and showing the plot
plt.title('Loss and Perplexity for Text Generation')
plt.show()
