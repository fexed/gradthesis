import matplotlib.pyplot as plot

fig, ax = plot.subplots()
#plot.figure(figsize=(20, 3))
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
ax.set_xticklabels(['Naive', 'Cumul.', 'Repl.', 'Epis.\nAccuratezza', 'EWC', 'LWF', '', 'Naive', 'Cumul.', 'Repl.', 'Epis.\nBWT', 'EWC', 'LWF', '', 'Naive', 'Cumul.', 'Repl.', 'Epis.\nFWT', 'EWC', 'LWF'])
ax.set_ylabel("Percentuale")
ax.set_ylim([-0.06, 1])
ax.set_title("Risultati in media")
ax.bar(1, 0.6586, color="orange")
ax.bar(2, 0.6956, color="green")
ax.bar(3, 0.6830, color="red")
ax.bar(4, 0.6770, color="purple")
ax.bar(5, 0.6472, color="olive")
ax.bar(6, 0.6390, color="cyan")
ax.bar(8, -0.0022, color="orange")
ax.bar(9, 0.0493, color="green")
ax.bar(10, -0.0057, color="red")
ax.bar(11, 0.0412, color="purple")
ax.bar(12, -0.0087, color="olive")
ax.bar(13, 0.0148, color="cyan")
ax.bar(15, 0.1887, color="orange")
ax.bar(16, 0.1918, color="green")
ax.bar(17, 0.2709, color="red")
ax.bar(18, 0.1808, color="purple")
ax.bar(19, 0.1847, color="olive")
ax.bar(20, 0.0648, color="cyan")
plot.gca().margins(x=0)
plot.gcf().canvas.draw()
tl = plot.gca().get_xticklabels()
maxsize = max([t.get_window_extent().width for t in tl])
m = 0.25 # inch margin
s = maxsize/plot.gcf().dpi*20+1.5*m
margin = m/plot.gcf().get_size_inches()[0]

plot.gcf().subplots_adjust(left=margin, right=1.01-margin)
plot.gcf().set_size_inches(s, plot.gcf().get_size_inches()[1])
plot.savefig("plots/mean_final_metrics.png")
