import matplotlib.pyplot as plot

fig, ax = plot.subplots()
#plot.figure(figsize=(20, 3))
ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax.set_xticklabels(['Off.', 'Cont.', 'Cumul.', 'Repl.', 'Ep.', 'EWC', 'LWF'])
ax.set_ylabel("Accuracy")
ax.set_ylim([0, 100])
ax.bar(1, 87.77, yerr=0, color="blue")
ax.bar(2, 87.01, yerr=0.64, color="orange")
ax.bar(3, 87.66, yerr=0.16, color="green")
ax.bar(4, 87.13, yerr=0.75, color="red")
ax.bar(5, 82.19, yerr=3.95, color="purple")
ax.bar(6, 87.34, yerr=0.42, color="olive")
ax.bar(7, 86.91, yerr=0.89, color="cyan")
plot.gca().margins(x=0)
plot.gcf().canvas.draw()
tl = plot.gca().get_xticklabels()
maxsize = max([t.get_window_extent().width for t in tl])
m = 0.5 # inch margin
s = maxsize/plot.gcf().dpi*7+2*m
margin = m/plot.gcf().get_size_inches()[0]

plot.gcf().subplots_adjust(left=margin, right=1.-margin)
plot.gcf().set_size_inches(s, plot.gcf().get_size_inches()[1])
plot.savefig("plots/customascertain_final_accuracy.png")
