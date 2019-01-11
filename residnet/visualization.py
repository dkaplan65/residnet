import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import copy

import sys
sys.path.append('..')
from .constants import *

def plot_errors_relative(ax, arr_primary, lst_arrs, lst_labels,
	label_primary = None, primary_color = None, colors = None,
	linewidth = None, **kwargs):

	'''Plots the errors of the arrays in `lst_arrs` in respective order to
	the sorted `arr`

	Throws error if the length of every array being plotted are not the
	same length
	--------------
	args
	--------------
	ax (matplotlib.pyplot.Axes)
		- Axes object to plot on
	arr_primary (1-dim array)
		- Array of errors
		- This is the array that everything is reshuffled relative to
	label_primary (str)
		- Label for `arr_primary`
	lst_arrs (list(1-dim array))
		- a list of 1-dim arrays that you are reshuffling with respect
		  to `arr_primary`
	lst_labels (list(str))
		- The label for each respective array in `lst_arrays`
		- Throws error if len(lst_arrs) != len(lst_labels)
	color, primary_color (list)
		- Colors of the lines
		- If None, do not specify
	linewidth (float)
		- Linewidth
	**kwargs (dict)
		- optional attributes for labeling
	'''

	for arr_ in lst_arrs:
		if len(arr_) != len(arr_primary):
			raise ValueError('plot_errors_relative: All the arrays are not the same length')
	if len(lst_arrs) != len(lst_labels):
		raise ValueError('plot_errors_relative: Length of other arrays and labels must be the same')
	if primary_color is None:
		primary_color = DEFAULT_VIS_PLOT_REL_PRIM_COLOR
	if linewidth == None:
		linewidth = DEFAULT_VIS_PLOT_LINEWIDTH
	if label_primary == None:
		label_primary = DEFAULT_VIS_PRIMARY_LABEL

	# Start plotting
	idxs = np.argsort(arr_primary)
	x = np.arange(len(idxs))
	for i in range(len(lst_arrs)):
		if colors == None:
			color = None
		else:
			color = colors[i]
		arr_ = lst_arrs[i]
		ax.plot(
			x = x,
			y = arr_[idx],
			label = lst_labels[i],
			color = color,
			linewidth = linewidth,
			alpha = 0.5)

	# Plot primary label last so that it is on top
	ax.plot(
		x = x,
		y = arr_primary[idxs],
		color = primary_color,
		linewidth = linewidth,
		label = label_primary)

	return set_ax_labels(ax)

def map(ax, arr, cmap = None, vmin = None, vmax = None, **kwargs):
	'''Plots a map for the given array.
	Assumes that the given array is already in the right orientation.
	-----------
	args
	-----------
	arr (np.ndarray)
		- 2-dim array we are creating an image from
	cmap (matplotlib.cm)
		- colormap
	'''
	im = ax.imshow(arr, cmap = cmap, vmin=vmin, vmax=vmax)

	# Remove ticks
	ax.tick_params(
		axis = 'both',
		which = 'both',
		bottom = False,
		top = False,
		left = False,
		right = False,
		labelbottom = False,
		labeltop = False,
		labelleft = False,
		labelright = False)
	set_ax_labels(ax,**kwargs)
	return im

def map_over_time(src, cmap, **kwargs):
	'''Maps the DataWrapper contents for each day in the year
	and saves each day as a file. They are then stitched together in
	an mp4. Iteratively calls `visualization.map` and
	`data_processing.transforms.mapify`.

	Useful for things like:
		- classification error over time
		- interpolation error over time
		- subgrid classification over time
		- etc.

	src (data_processing.wrappers.DataWrapper)
		- Data
	'''
	raise NotImplemented

def set_ax_labels(ax, title = None, title_size = None, xlabel = None,
	xlabel_size = None, ylabel = None, ylabel_size = None):
	'''
	Adds optional labels to the specified Axes
	'''
	if title is not None:
		if title_size is not None:
			ax.set_title(title, fontsize = title_size)
		else:
			ax.set_title(title)
	if xlabel is not None:
		if xlabel_size is not None:
			ax.set_xlabel(xlabel, fontsize = xlabel_size)
		else:
			ax.set_xlabel(xlabel)
	if ylabel is not None:
		if ylabel_size is not None:
			ax.set_ylabel(ylabel, fontsize = ylabel_size)
		else:
			ax.set_ylabel(ylabel)
	return ax
