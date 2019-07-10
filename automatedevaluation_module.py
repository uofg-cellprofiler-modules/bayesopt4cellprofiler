# coding=utf-8

"""
Author: Lisa Laux, MSc Information Technology, University of Glasgow
Date: August 2018
Updated: May 2019

License: Please note that the CellProfiler Software was released under the BSD 3-Clause License by
the Broad Institute: Copyright © 2003 - 2018 Broad Institute, Inc. All rights reserved. Please refer to
CellProfiler's LICENSE document for details.

"""

#################################
#
# Imports from useful Python libraries
#
#################################

import numpy
import skimage.color
import skimage.segmentation
import skimage.util

#################################
#
# Imports from CellProfiler
#
#################################

import cellprofiler.image
import cellprofiler.module
import cellprofiler.measurement
import cellprofiler.object
import cellprofiler.setting
import cellprofiler.pipeline
import cellprofiler.workspace

__doc__ = """\
AutomatedEvaluation
===================

**AutomatedEvaluation** can be used to automatically evaluate the quality of identified objects
(eg nuclei, cytoplasm, adhesions). 

By choosing an object and some of its measurements to be evaluated, the module will check whether these measurements
are in a tolerance range provided in the settings. If object measurement values are outside this range, a 
percentaged deviation per value will be measured and saved with the object measurements in a numpy array.
The module also displays the object's outlines and supporting objects if display output is enabled.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============


Requirements
^^^^^^^^^^^^
The module needs to be placed after both, IdentifyObjects and Measurement modules to access the objects and
measurements taken for these objects to evaluate them as input.


Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Evaluation:**

-  *Evaluation_Deviation*: Array of percentaged deviations of each object to a pre-defined quality range.
        e.g.    the deviation of a measurement with value 0.5 to the min. quality threshold of 0.9 will be 44.4 (%)
                the deviation of a measurement with value 1.1 to the max. quality thresthold of 1.0 will be 10.0 (%)

"""

#
# Constants
#
CATEGORY = 'Evaluation'
DEVIATION = 'Deviation'
FEATURE_NAME = 'Evaluation_Deviation'

NUM_FIXED_SETTINGS = 5
NUM_GROUP1_SETTINGS = 2
NUM_GROUP2_SETTINGS = 2

COLORS = {"White": (1, 1, 1),
          "Black": (0, 0, 0),
          "Red": (1, 0, 0),
          "Green": (0, 1, 0),
          "Blue": (0, 0, 1),
          "Yellow": (1, 1, 0)}

COLOR_ORDER = ["Red", "Green", "Blue", "Yellow", "White", "Black"]


#
# Create module class which inherits from cellprofiler.module.Module class
#
class AutomatedEvaluation(cellprofiler.module.Module):

    #
    # Declare the name for displaying the module, e.g. in menus 
    # Declare the category under which it is stored and grouped in the menu
    # Declare variable revision number which can be used to provide backwards compatibility  if CellProfiler will be
    # released in a new version
    #
    module_name = "AutomatedEvaluation"
    category = "Advanced"
    variable_revision_number = 1

    #######################################################################
    # Create and set CellProfiler settings for GUI and Pipeline execution #
    #######################################################################

    #
    # Define the setting's data types and grouped elements
    #
    def create_settings(self):
        super(AutomatedEvaluation, self).create_settings()

        module_explanation = [
            "Module used to automatically evaluate the quality of identified objects (eg nuclei, adhesions). "
            "Needs to be placed after IdentifyObjects and Measurement modules. Choose the object which measurements "
            "shall be evaluated first. You may choose more objects as well that will be outlined and displayed. "
            "Then choose the measurements to be evaluated and set a tolerance range for their values. If objects are "
            "outside this range, a deviation will be measured and saved with the object."]

        #
        # Notes will appear in the notes-box of the module
        #
        self.set_notes([" ".join(module_explanation)])

        #
        # ImageNameSubscriber provides all available images in the image set
        # The image is needed to display the outlines of an object on the image to the user
        #
        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Select image on which to display outlines",
            cellprofiler.setting.NONE,
            doc="""\
        Choose the image to serve as the background for the outlines. You can
        choose from images that were loaded or created by modules previous to
        this one.
        """
        )

        #
        # Choose a mode for outlining the objects on the image
        #
        self.line_mode = cellprofiler.setting.Choice(
            "How to outline",
            ["Inner", "Outer", "Thick"],
            value="Inner",
            doc="""\
        Specify how to mark the boundaries around an object:

        -  *Inner:* outline the pixels just inside of objects, leaving
           background pixels untouched.
        -  *Outer:* outline pixels in the background around object boundaries.
           When two objects touch, their boundary is also marked.
        -  *Thick:* any pixel not completely surrounded by pixels of the same
           label is marked as a boundary. This results in boundaries that are 2
           pixels thick.
        """
        )

        #
        # Provide a name for the created output image (which can be saved)
        #
        self.output_image_name = cellprofiler.setting.ImageNameProvider(
            "Name the output image",
            "AutoEvaluationOverlay",
            doc="""\
        Enter the name of the output image with the outlines overlaid. This
        image can be selected in later modules (for instance, **SaveImages**).
        """
        )

        self.spacer = cellprofiler.setting.Divider(line=False)

        #
        # The number of outlined objects;
        # necessary for prepare_settings method
        #
        self.count1 = cellprofiler.setting.Integer(
            'No. of objects to display',
            1,
            minval=1,
            maxval=100,
            doc="""\
        No. of outlined objects."""
        )

        #
        # The number of measurements for the object (first one in outlines);
        # necessary for prepare_settings method
        #
        self.count2 = cellprofiler.setting.Integer(
            'No. of measurements to consider for object',
            1,
            minval=1,
            maxval=100,
            doc="""\
        No. of measurements."""
        )

        #
        # Group of outlines for different object that should be displayed on the image
        #
        self.outlines = []

        #
        # Add first outline which cannot be deleted; there must be at least one
        #
        self.add_outline(can_remove=False)

        #
        # Button for adding additional outlines; calls add_outline helper function
        #
        self.add_outline_button = cellprofiler.setting.DoSomething("", "Add another outline", self.add_outline)

        self.divider = cellprofiler.setting.Divider()

        #
        # Group of measurements made for the object by a Measurements module
        #
        self.measurements = []

        #
        # Add first measurement which cannot be deleted; there must be at least one
        #
        self.add_measurement(can_delete=False)

        #
        # Button for adding additional measurements; calls add_measurement helper function
        #
        self.add_measurement_button = cellprofiler.setting.DoSomething(
            "", "Add another measurement", self.add_measurement)

    #
    # helper function:
    # add objects to outline and color chooser
    # add a remove-button for all outlines except a mandatory one
    #

    def add_outline(self, can_remove=True):
        group = cellprofiler.setting.SettingsGroup()
        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=False))

        #
        # Object to be outlined which was identified in upstream IndentifyObjects module;
        # accessible via ObjectNameSubscriber
        #
        group.append(
            "objects_name",
            cellprofiler.setting.ObjectNameSubscriber(
                "Select objects to display",
                cellprofiler.setting.NONE,
                doc="Choose the objects whose outlines you would like to display. The first object chosen will be the "
                    "leading object, storing the quality measurement needed for the Bayesian Optimisation."
            )
        )

        default_color = (COLOR_ORDER[len(self.outlines)] if len(self.outlines) < len(COLOR_ORDER) else COLOR_ORDER[0])

        #
        # Color chooser for setting the outline color to display on the image
        #
        group.append(
            "color",
            cellprofiler.setting.Color(
                "Select outline color",
                default_color,
                doc="Objects will be outlined in this color."
            )
        )

        if can_remove:
            group.append(
                "remover",
                cellprofiler.setting.RemoveSettingButton("", "Remove this outline", self.outlines, group)
            )

        self.outlines.append(group)

    #
    # helper function:
    # adds measurements grouped with corresponding quality ranges
    # adds a remove-button for all measurements except a mandatory one where can_delete = False
    #
    def add_measurement(self, can_delete=True):
        group = cellprofiler.setting.SettingsGroup()

        #
        # Dropdown selection for measurements taken for the object
        #
        group.append(
            "measurement",
            cellprofiler.setting.Measurement(
                "Select the quality measurement",
                self.outlines[0].objects_name.get_value,
                "AreaShape_Area",
                doc="""\
See the **Measurements** modules help pages for more information
on the features measured."""

            )
        )

        #
        # Range of values which are within the accepted quality thresholds
        #
        group.append(
            "range",
            cellprofiler.setting.FloatRange(
                'Set tolerance range',
                (00.00, 100.00),
                minval=00.00,
                maxval=100000.00,
                doc="""\
Set a tolerance range for the measurement. If values of the measurement are not within the range, a percentaged 
deviation will be calculated"""
            )
        )

        group.append("divider", cellprofiler.setting.Divider())

        self.measurements.append(group)

        if can_delete:
            group.append(
                "remover",
                cellprofiler.setting.RemoveSettingButton(
                    "",
                    "Remove this measurement",
                    self.measurements, group
                )
            )

    #
    # setting_values are stored as unicode strings in the pipeline.
    # If the module has settings groups, it needs to be ensured that settings() returns the correct
    # number of settings as saved in the file.
    # To do so, look at the setting values before settings() is called to determine how many to return.
    # Add groups if necessary.
    #
    def prepare_settings(self, setting_values):

        #
        # No. of outlines in outlines group
        #
        count1 = int(setting_values[3])

        #
        # No. of measurements in measurement group
        #
        count2 = int(setting_values[4])

        #
        # Handle adding outlines
        #
        num_settings_1 = (len(setting_values) - NUM_FIXED_SETTINGS - NUM_GROUP2_SETTINGS*count2) / NUM_GROUP1_SETTINGS

        if len(self.outlines) == 0:
            self.add_outline(False)
        elif len(self.outlines) > num_settings_1:
            del self.outlines[num_settings_1:]
        else:
            for i in range(len(self.outlines), num_settings_1):
                self.add_outline()

        #
        # Handle adding measurements
        #
        num_settings_2 = (len(setting_values) - NUM_FIXED_SETTINGS - NUM_GROUP1_SETTINGS * count1) / NUM_GROUP2_SETTINGS

        if len(self.measurements) == 0:
            self.add_measurement(False)
        elif len(self.measurements) > num_settings_2:
            del self.measurements[num_settings_2:]
        else:
            for i in range(len(self.measurements), num_settings_2):
                self.add_measurement()

    #  
    # CellProfiler must know about the settings in the module.
    # This method returns the settings in the order that they will be loaded and saved from a pipeline or project file.
    # Accessing setting members of a group of settings requires looping through the group result list
    #
    def settings(self):
        result = [self.image_name, self.output_image_name, self.line_mode]
        result += [self.count1, self.count2]
        for outline in self.outlines:
            result += [outline.color, outline.objects_name]
        for measurement in self.measurements:
            result += [measurement.measurement, measurement.range]
        return result

    #  
    # returns what the user should see in the GUI
    # include buttons and dividers which are not added in the settings method
    #
    def visible_settings(self):
        result = [self.image_name, self.line_mode, self.spacer]
        result += [self.count1]
        for outline in self.outlines:
            if hasattr(outline, "divider"):
                result += [outline.divider]
            result += [outline.objects_name, outline.color]
            if hasattr(outline, "remover"):
                result += [outline.remover]
        result += [self.add_outline_button, self.divider]
        result += [self.count2]
        for measurement in self.measurements:
            result += [measurement.measurement, measurement.range]
            if hasattr(measurement, "remover"):
                result += [measurement.remover]
            if hasattr(measurement, "divider"):
                result += [measurement.divider]
        result += [self.add_measurement_button]
        return result

    ###################################################################
    # Run method will be executed in a worker thread of the pipeline #
    ###################################################################

    #
    # CellProfiler calls "run" on each image set in the pipeline
    # The workspace as input parameter contains the state of the analysis so far
    #
    def run(self, workspace):

        #
        # Get the measurements object for the current run
        #
        workspace_measurements = workspace.measurements

        #
        # Get the image pixels from the image set
        #
        base_image, dimensions = self.base_image(workspace)

        #
        # get the object outlines as pixel data
        #
        pixel_data = self.run_color(workspace, base_image.copy())

        #
        # create new output image with the object outlines
        #
        output_image = cellprofiler.image.Image(pixel_data, dimensions=dimensions)

        #
        # add new image with object outlines to workspace image set
        #
        workspace.image_set.add(self.output_image_name.value, output_image)
        image = workspace.image_set.get_image(self.image_name.value)

        #
        # set the input image as the parent image of the output image
        #
        output_image.parent_image = image

        #
        # declare array of deviation values
        #
        deviations = []

        #
        # take the selected object's measurements and corresponding ranges one by one and compare it to the thresholds
        #
        for m in self.measurements:

            p_dev = 0  # percentaged deviation of the measurement m

            #
            # get_current_measurement returns an array with the measurements per object
            #
            measurement_values = workspace_measurements.get_current_measurement(
                self.outlines[0].objects_name.value_text, m.measurement.value_text)
            # print("All measurements:")
            # print(measurement_values)

            #
            # loop through all entries of the array to determine whether they are within the quality range or not;
            # if not, determine the deviation to the min or max bound respectively and save the percentaged deviation
            # in deviations
            #
            for v in measurement_values:
                if v >= m.range.min and v <= m.range.max:
                    p_dev += 0
                else:
                    if v < m.range.min:
                        deviation = m.range.min - v
                        p_dev += (deviation*100)/m.range.min
                        # print("p_dev:")
                        # print(p_dev)
                    else:
                        deviation = v - m.range.max
                        p_dev += (deviation * 100) / m.range.max
                        # print("p_dev:")
                        # print(p_dev)

            #
            # calculate the percentaged value in relation to all values in the array to weight it proportionally and add
            # it to deviations
            #
            if len(measurement_values) == 0:
                deviations += [0]
            else:
                deviations += [p_dev/len(measurement_values)]

        #
        # transform deviations to a numpy array after all separate deviations for different measurements m have been
        # collected
        #
        dev_array = numpy.array(deviations)

        # print(dev_array)

        #
        # Add measurement for deviations to workspace measurements to make it available to downstream modules,
        # e.g. the Bayesian Module
        #
        workspace.add_measurement(self.outlines[0].objects_name.value, FEATURE_NAME, dev_array)

        #
        # if user wants to show the display-window, save data needed for display in workspace.display_data
        #
        if self.show_window:
            workspace.display_data.pixel_data = pixel_data

            workspace.display_data.image_pixel_data = base_image

            workspace.display_data.dimensions = dimensions

    #
    # if user wants to show the display window during pipeline execution, this method is called by UI thread
    # display the data saved in display_data of workspace
    # used a CP defined figure to plot/display data via matplotlib
    #

    def display(self, workspace, figure):
        #
        # show outlined image
        #
        dimensions = workspace.display_data.dimensions

        figure.set_subplots((2, 1), dimensions=dimensions)

        figure.subplot_imshow_bw(
            0,
            0,
            workspace.display_data.image_pixel_data,
            self.image_name.value
        )

        figure.subplot_imshow(
            1,
            0,
            workspace.display_data.pixel_data,
            self.output_image_name.value,
            sharexy=figure.subplot(0, 0)
        )

    #
    # helper method;
    # Gets the image pixels from the image in the workspace
    #
    def base_image(self, workspace):

        image = workspace.image_set.get_image(self.image_name.value)

        pixel_data = skimage.img_as_float(image.pixel_data)

        if image.multichannel:
            return pixel_data, image.dimensions

        return skimage.color.gray2rgb(pixel_data), image.dimensions

    #
    # helper method;
    # prepares colors to draw the outlines of the objects selected
    #
    def run_color(self, workspace, pixel_data):
        for outline in self.outlines:
            objects = workspace.object_set.get_objects(outline.objects_name.value)

            color = tuple(c / 255.0 for c in outline.color.to_rgb())

            pixel_data = self.draw_outlines(pixel_data, objects, color)

        return pixel_data

    #
    # helper method;
    # draws the outlines of the objects selected
    #
    def draw_outlines(self, pixel_data, objects, color):
        for labels, _ in objects.get_labels():
            resized_labels = self.resize(pixel_data, labels)

            if objects.volumetric:
                for index, plane in enumerate(resized_labels):
                    pixel_data[index] = skimage.segmentation.mark_boundaries(
                        pixel_data[index],
                        plane,
                        color=color,
                        mode=self.line_mode.value.lower()
                    )
            else:
                pixel_data = skimage.segmentation.mark_boundaries(
                    pixel_data,
                    resized_labels,
                    color=color,
                    mode=self.line_mode.value.lower()
                )

        return pixel_data

    #
    # helper method;
    # resizes the object labels
    #
    def resize(self, pixel_data, labels):
        initial_shape = labels.shape

        final_shape = pixel_data.shape

        if pixel_data.ndim > labels.ndim:
            final_shape = final_shape[:-1]

        adjust = numpy.subtract(final_shape, initial_shape)

        cropped = skimage.util.crop(
            labels,
            [(0, dim_adjust) for dim_adjust in numpy.abs(numpy.minimum(adjust, numpy.zeros_like(adjust)))]
        )

        return numpy.pad(
            cropped,
            [(0, dim_adjust) for dim_adjust in numpy.maximum(adjust, numpy.zeros_like(adjust))],
            mode="constant",
            constant_values=(0)
        )

    #
    # helper method;
    # determines 3D
    #
    def volumetric(self):
        return True

    ####################################################################
    # Tell CellProfiler about the measurements produced in this module #
    ####################################################################

    #
    # Provide the measurements for use in the database or a spreadsheet
    #
    def get_measurement_columns(self, pipeline):

        input_object_name = self.outlines[0].objects_name.value

        return [input_object_name, FEATURE_NAME, cellprofiler.measurement.COLTYPE_FLOAT]

    #
    # Return a list of the measurement categories produced by this module if the object_name matches
    #
    def get_categories(self, pipeline, object_name):
        if object_name == self.outlines[0].objects_name:
            return [CATEGORY]

        return []

    #
    # Return the feature names if the object_name and category match to the GUI for measurement subscribers
    #
    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.outlines[0].objects_name and category == CATEGORY:
            return [DEVIATION]

        return []





