from tkinter import *
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
from scipy import stats
import cv2
from predictions import res,actions,image

# plot function is created for
# plotting the graph in
# tkinter window
colors = [(245,117,16), (117,245,16), (16,117,245)]
prob_viz =(res, actions, input_frame, colors):

def plot():

	fig = Figure(figsize=(18,18))
	plt.imshow(prob_viz(res, actions, image, colors))

	# creating the Tkinter canvas
	# containing the Matplotlib figure
	canvas = FigureCanvasTkAgg(fig,
							master = window)
	canvas.draw()

	# placing the canvas on the Tkinter window
	canvas.get_tk_widget().pack()

	# creating the Matplotlib toolbar
	toolbar = NavigationToolbar2Tk(canvas,
								window)
	toolbar.update()

	# placing the toolbar on the Tkinter window
	canvas.get_tk_widget().pack()

# the main Tkinter window
window = Tk()

# setting the title
window.title('Plotting in Tkinter')

# dimensions of the main window
window.geometry("700x700")

# button that displays the plot
plot_button = Button(master = window,
					command = plot,
					height = 2,
					width = 10,
					text = "Plot")

# place the button
# in main window
plot_button.pack()

# run the gui
window.mainloop()
