# coding=utf-8

"""
Author: Lisa Laux, MSc Information Technology, University of Glasgow
Date: August 2018
Updates: May 2019

License: Please note that the CellProfiler Software was released under the BSD 3-Clause License by
the Broad Institute: Copyright © 2003 - 2018 Broad Institute, Inc. All rights reserved. Please refer to
CellProfiler's LICENSE document for details.

"""

#################################
#
# Imports from useful Python libraries
#
#################################

import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from copy import deepcopy
from itertools import product
import os

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
import cellprofiler.preferences


__doc__ = """\
BayesianOptimisation
===================

**BayesianOptimisation** uses Bayesian Optimisation methods on parameters (settings) chosen from modules placed before 
this module in the pipeline. It needs either a ManualEvaluation and/ or AutomatedEvaluation placed upstream 
in the pipeline. It can only evaluate and operate on the quality measurements of one object at a time. 

The Bayesian Optimisation will only be executed if required quality thresholds/ranges defined in the evaluation 
module(s) are not met.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES           NO
============ ============ ===============


Requirements
^^^^^^^^^^^^
There must be at least one evaluation module, either **ManualEvaluation** or **AutomatedEvaluation** placed
before this module in order to have evaluation measurements available.
Either only one or both evaluation measurements can be chosen to be taken as quality measure for the optimisation 
procedure.


Technical notes
^^^^^^^^^^^^^^
The max. number of parameters to optimise is currently 4. 

There is a filter set for only making parameters form the IdentifyObjects modules available for optimisation. This 
can be changed by just removing the filter in the get_module_list helper method.


References
^^^^^^^^^^
The basic code for the Bayesian Optimisation procedure was provided by Bjørn Sand Jensen (University of Glasgow) 
-- bjorn.jensen@glasgow.ac.uk -- and altered for the purposes of this module.

"""

#
# Constants
#
NUM_FIXED_SETTINGS = 9
NUM_GROUP1_SETTINGS = 1
NUM_GROUP2_SETTINGS = 4

#
# for testing/ printout purposes only
#
np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=np.nan)
# import pdb
# import pdbi


#
# Create module class which inherits from cellprofiler.module.Module class
#
class BayesianOptimisation(cellprofiler.module.Module):

    #
    # Declare the name for displaying the module, e.g. in menus 
    # Declare the category under which it is stored and grouped in the menu
    # Declare variable revision number which can be used to provide backwards compatibility  if CellProfiler will be
    # released in a new version
    #
    module_name = "BayesianOptimisation"
    category = "Advanced"
    variable_revision_number = 1

    #######################################################################
    # Create and set CellProfiler settings for GUI and Pipeline execution #
    #######################################################################

    #
    # Define the setting's data types and grouped elements
    #
    def create_settings(self):
        module_explanation = [
            "This module uses BayesianOptimisation on parameters (settings) chosen from modules placed before this "
            "module in the pipeline. Step 1: Choose the object which you have evaluated in the evaluation modules. "
            "The Bayesian module considers these measures as quality indicators. Step 2: Choose the parameters "
            "(settings) to be adjusted. Bayesian Optimisation will be executed if required quality thresholds/ranges "
            "are not met. If an AutomatedEvaluation modue is placed beforehand, the final output of the autoevaluation"
            "will be displayed and can be evaluated by the user."]

        #
        # Notes will appear in the notes-box of the module
        #
        self.set_notes([" ".join(module_explanation)])

        #
        # Object identified in upstream IndentifyObjects module; accessible via ObjectNameSubscriber
        #
        self.input_object_name = cellprofiler.setting.ObjectNameSubscriber(
            "Input object name",
            cellprofiler.setting.NONE,
            doc="""\
These are the objects that the module operates on."""
        )

        #
        # The number of evaluation modules as input for BayesianModule;
        # necessary for prepare_settings method
        #
        self.count1 = cellprofiler.setting.Integer(
                'No. of evaluation modules',
                1,
                minval=1,
                maxval=2,
                doc="""\
No. of evaluation modules before BayesianModule."""
            )

        #
        # The number of parameters to be adjusted by BayesianModule;
        # necessary for prepare_settings method
        #
        self.count2 = cellprofiler.setting.Integer(
            'No. of settings to be adjusted',
            2,
            minval=1,
            maxval=4,
            doc="""\
No. of settings that should be adjusted by BayesianModule. You can choose up to 4 settings."""
        )

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

        self.spacer = cellprofiler.setting.Divider(line=True)

        #
        # The weighting in % for the automated evaluation results
        #
        self.weighting_auto = cellprofiler.setting.Integer(
            'Weighting of automated evaluation score (%)',
            50,
            minval=0,
            maxval=100,
            doc="""\
The weighting of the automated evaluation results in comparison to manual evaluation."""
        )

        #
        # The weighting in % for the manual evaluation results
        #
        self.weighting_manual = cellprofiler.setting.Integer(
            'Weighting of manual evaluation score (%)',
            50,
            minval=0,
            maxval=100,
            doc="""\
The weighting of the manual evaluation result in comparison to automated evaluation."""
        )


        self.spacer6 = cellprofiler.setting.Divider(line=True)

        #
        # The maximum number of iterations for the Bayesian Optimisation
        #
        self.max_iter = cellprofiler.setting.Integer(
            'Max. iterations for Bayesian Optimisation',
            150,
            minval=2,
            maxval=10000,
            doc="""\
Define the maximum number of iterations the Bayesian Optimisation should run. The minimum number is 2, 
recommended iterations are 50 - 200, depending on the problem to be solved. """
        )

        #
        # The length scale for the Bayesian Optimisation kernel function
        #
        self.length_scale = cellprofiler.setting.Float(
            'Length scale for Bayesian Optimisation kernel function',
            0.1,
            minval=0,
            maxval=1000,
            doc="""\
Define the length scale for the kernel function of the Bayesian Optimisation model. The value influences the
smoothness of the objective kernel function. A larger value indicates a smoother function."""
        )

        #
        # The alpha value for the Bayesian Optimisation model
        #
        self.alpha = cellprofiler.setting.Float(
            'Alpha for Bayesian Optimisation model',
            0.01,
            minval=0,
            maxval=1000,
            doc="""\
Define the alpha value for the GaussianProcessRegressor model. A low value indicates low noise in the data."""
        )

        self.spacer4 = cellprofiler.setting.Divider(line=True)

        self.parameters = []

        #
        # Add first parameter which cannot be deleted; there must be at least one
        #
        self.add_parameter(can_remove=False)

        #
        # Button for adding additional parameters; calls add_parameter helper function
        #
        self.add_param_button = cellprofiler.setting.DoSomething("", "Add parameter", self.add_parameter)

        self.spacer2 = cellprofiler.setting.Divider(line=True)

        #
        # Button for refreshing the GUI; calls refreshGUI helper function
        # This is necessary as the choices_fn function does not work without
        # refreshing the GUI if new groups were added
        #
        self.refresh_button = cellprofiler.setting.DoSomething(
            "",
            "Refresh GUI",
            self.refreshGUI,
            doc = """\
If the dropdown menus are not updated, you can update them again with this button."""
        )

        self.spacer3 = cellprofiler.setting.Divider(line=True)

        #
        # Output directory chooser;
        # x and y values of previous Optimisation rounds will be saved in files in this directory
        #
        self.pathname = cellprofiler.setting.DirectoryPath(
            "Output file location",
            dir_choices=[
                cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
                cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME,
                cellprofiler.preferences.ABSOLUTE_FOLDER_NAME,
                cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                cellprofiler.preferences.DEFAULT_INPUT_SUBFOLDER_NAME],
            doc="""\
Choose the directory where Optimisation data is saved. """
        )

        self.spacer5 = cellprofiler.setting.Divider(line=False)

        #
        # Button for deleting existing files storing values from previous runs
        #
        self.delete_button = cellprofiler.setting.DoSomething(
            "",
            "Delete previous Data",
            self.delete_data,
            doc="""\
If there is previously gathered data saved in a file you can choose to delete it."""
        )

    #
    # helper function:
    # add the quality measurements which should be considered by B.O.
    # add a remove-button for all measurements except a mandatory one
    #
    def add_measurement(self, can_delete=True):
        group = cellprofiler.setting.SettingsGroup()

        #
        # Dropdown selection for measurements taken for the object
        #
        group.append(
            "evaluation_measurement",
            cellprofiler.setting.Measurement(
                "Select measurements for evaluation",
                self.input_object_name.get_value,
                "Evaluation_Deviation",
                doc="""\
See the **Measurements** modules help pages for more information on the
features measured."""

            )
        )

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
    # helper function:
    # add parameters grouped with corresponding modules
    # add a remove-button for all parameters except a mandatory one
    #
    def add_parameter(self, can_remove=True):

        group = cellprofiler.setting.SettingsGroup()

        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=False))

        #
        # Dropdown selection for modules (IdentifyObjects modules)
        #
        group.append("module_names", cellprofiler.setting.Choice(
            "Select module",
            choices=[""],
            choices_fn=self.get_module_list,
            doc="""\
This is the module where Bayesian Optimisation will adjust settings
"""
        ))

        #
        # Dropdown selection for parameters of the selected modules
        #
        group.append("parameter_names", cellprofiler.setting.Choice(
            "Select parameter",
            choices=[""],
            choices_fn=self.get_settings_from_modules,
            doc="""\
These are the settings to be adjusted by Bayesian Optimisation
"""
        ))

        #
        # The parameters will be adjusted within this range
        #
        group.append(
            "range",
            cellprofiler.setting.FloatRange(
                'Set min and max boundaries for variation',
                (1.00, 100.00),
                minval=00.00,
                maxval=1000.00,
                doc="""\
The Bayesian Optimisation will vary the parameter within this range of candidates. Please note that the lower
bound is inclusive, the upper bound is exclusive."""
            )
        )

        #
        # The parameters will be adjusted within this range
        #
        group.append(
            "steps",
            cellprofiler.setting.Float(
                'Set steps between boundaries',
                0.1,
                minval=00.00,
                maxval=10.00,
                doc="""\
The variation steps within the chosen range for choosing a candidate set."""
            )
        )

        if can_remove:
            group.append("remover",
                         cellprofiler.setting.RemoveSettingButton("", "Remove parameter", self.parameters, group))

        self.parameters.append(group)

    #
    # setting_values are stored as unicode strings in the pipeline file.
    # If the module has settings groups, it needs to be ensured that settings() returns the correct
    # number of settings as saved in the file.
    # To do so, look at the number of setting values before settings() is called to determine how many to return.
    # Add groups if necessary.
    #
    def prepare_settings(self, setting_values):

        #
        # No. of measurements in measurements group
        #
        count1 = int(setting_values[1])

        #
        # No. of parameters in parameters group
        #
        count2 = int(setting_values[2])

        #
        # Handle adding measurements
        #
        num_settings_1 = (len(setting_values) - NUM_FIXED_SETTINGS - NUM_GROUP2_SETTINGS*count2) / NUM_GROUP1_SETTINGS

        if len(self.measurements) == 0:
            self.add_measurement(False)
        elif len(self.measurements) > num_settings_1:
            del self.measurements[num_settings_1:]
        else:
            for i in range(len(self.measurements), num_settings_1):
                self.add_measurement()

        #
        # Handle adding parameters
        #
        num_settings_2 = (len(setting_values) - NUM_FIXED_SETTINGS - NUM_GROUP1_SETTINGS * count1) / NUM_GROUP2_SETTINGS

        if len(self.parameters) == 0:
            self.add_parameter(False)
        elif len(self.parameters) > num_settings_2:
            del self.parameters[num_settings_2:]
        else:
            for i in range(len(self.parameters), num_settings_2):
                self.add_parameter()

    #  
    # CellProfiler must know about the settings in the module.
    # This method returns the settings in the order that they will be loaded and saved from a pipeline or project file.
    # Accessing setting members of a group of settings requires looping through the group result list
    #
    def settings(self):
        result = [self.input_object_name]
        result += [self.count1, self.count2]
        result += [self.weighting_manual, self.weighting_auto]
        for m in self.measurements:
            result += [m.evaluation_measurement]
        result += [self.max_iter, self.length_scale, self.alpha]
        for p in self.parameters:
            result += [p.module_names, p.parameter_names, p.range, p.steps]
        result += [self.pathname]

        return result

    #  
    # returns what the user should see in the GUI
    # include buttons and dividers which are not added in the settings method
    #
    def visible_settings(self):
        result = [self.input_object_name]
        result += [self.count1]
        for mod in self.measurements:
            result += [mod.evaluation_measurement]
            if hasattr(mod, "remover"):
                result += [mod.remover]
        result += [self.add_measurement_button, self.spacer, self.weighting_auto, self.weighting_manual, self.spacer6,
                   self.max_iter, self.length_scale, self.alpha, self.spacer4]
        result += [self.count2]
        for param in self.parameters:
            if hasattr(param, "divider"):
                result += [param.divider]
            result += [param.module_names, param.parameter_names, param.range, param.steps]
            if hasattr(param, "remover"):
                result += [param.remover]
        result += [self.add_param_button, self.spacer2, self.refresh_button,
                   self.spacer3, self.pathname, self.spacer5, self.delete_button]

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
        # create absolute pathname for data files to be saved
        #
        x_filename = "x_bo_{}.txt".format(self.get_module_num())
        y_filename = "y_bo_{}.txt".format(self.get_module_num())

        x_absolute_path = "{}/{}".format(self.pathname.get_absolute_path(), x_filename)
        y_absolute_path = "{}/{}".format(self.pathname.get_absolute_path(), y_filename)

        #
        # get the measurements made so far from workspace data
        #
        workspace_measurements = workspace.measurements

        #
        # get the pipeline object which saves the setting parameters
        #
        pipeline = workspace.get_pipeline()

        #
        # assume that optimisation is off
        #
        self.optimisation_on = False

        #
        # create empty lists to hold evaluation measurements
        #
        manual_evaluation_result = []
        auto_evaluation_results = []

        #
        # save the quality measurements and determine whether optimisation is needed or not
        #
        for m in self.measurements:

            if m.evaluation_measurement.value_text == "Evaluation_ManualQuality":
                manual_evaluation_result = workspace_measurements.get_current_measurement(
                    self.input_object_name.value, m.evaluation_measurement.value_text)
                for e in manual_evaluation_result:
                    if float(e) > 0.0:
                        self.optimisation_on = True

            elif m.evaluation_measurement.value_text == "Evaluation_Deviation":
                auto_evaluation_results = workspace_measurements.get_current_measurement(
                    self.input_object_name.value, m.evaluation_measurement.value_text)
                for e in auto_evaluation_results:
                    if float(e) > 0.0:
                        self.optimisation_on = True

        #
        # get modules and their settings
        #
        number_of_params = self.parameters.__len__()

        # save operational data in lists; the lists operate with indices;
        # an indices corresponds to a certain module, a setting name in this module and the value of this setting
        target_setting_module_list = []     # saves module objects
        target_setting_names_list = []      # saves setting names
        target_setting_values_list = []     # saves setting values of the selected settings in the module
        target_setting_range = []           # saves the ranges in which the setting values shall be manipulated
        target_setting_steps = []           # saves the steps the range can vary


        #
        # get the data for the lists by looping through all settings chosen by the user
        #
        for p in self.parameters:

            #
            # get the module object
            #
            name_list = p.module_names.value_text.split(" #")
            number = int(name_list[1])
            target_module = pipeline.module(number)

            #
            # save module number in module_list
            #
            target_setting_module_list += [number]

            #
            # get the setting name
            #
            target_setting_name = p.parameter_names.value_text

            for setting in target_module.settings():
                if setting.get_text() == target_setting_name:
                    #
                    # add setting name to Names_list and setting value to values_list
                    #
                    target_setting_names_list += [setting.get_text()]
                    target_setting_values_list += [setting.get_value()]

            #
            # save range and steps into lists; ranges are saved as a tuple
            #
            target_setting_range += [p.range.value]
            target_setting_steps += [float(p.steps.value)]


        #
        # start optimisation if quality is not satisfying
        #
        if self.optimisation_on:

            #
            # do the bayesian optimisation with a new function that takes the lists and returns new parameters for
            # the settings
            #
            new_target_settings_array, current_y_values = self.bayesian_optimisation(manual_evaluation_result,
                                                                                     auto_evaluation_results,
                                                                                     target_setting_values_list,
                                                                                     target_setting_range,
                                                                                     target_setting_steps,
                                                                                     number_of_params,
                                                                                     self.weighting_auto.value,
                                                                                     self.weighting_manual.value,
                                                                                     self.length_scale.value,
                                                                                     self.alpha.value)

            #
            # when the bayesian_optimisation method returns None, this indicates that max_iterations
            # are reached and B.O. is stopped
            #
            if new_target_settings_array is None:
                self.optimisation_on = False

                #
                # if user wants to show the display-window, save data needed for display in workspace.display_data
                #
                if self.show_window:
                    #
                    # we first need to search for the lowest available y and the corresponding X settings
                    #
                    x = np.loadtxt(x_absolute_path)
                    y = np.loadtxt(y_absolute_path)

                    ind_best = np.argmin(y)

                    #
                    # account for case with 1D array in first round and several dimension issues
                    #
                    try:
                        num_dimensions = x.ndim
                    except AttributeError:
                        num_dimensions = 0

                    if num_dimensions == 1 and number_of_params == 1:  # otherwise would also reshape 1st round of x 2D
                        x = x.reshape(-1, 1)
                    elif num_dimensions == 0:  # accounts for x 1D in first round
                        x = np.array([x])
                    elif np.size(y) == 1:
                        x = np.array([x])

                    x_best = x[ind_best]
                    x_best = x_best.flatten()

                    workspace.display_data.statistics = []
                    for i in range(number_of_params):
                        workspace.display_data.statistics.append(
                            (target_setting_names_list[i], x_best[i], target_setting_values_list[i]))

                    workspace.display_data.col_labels = ("Setting Name", "Best Value so far", "Final Value")
                    workspace.display_data.stop_info = "Max. number of iterations reached. Optimisation stopped."

            #
            # adjust the module settings with new x parameters returned form B.O.
            #
            else:
                new_target_settings = new_target_settings_array.flatten()
                current_y_values = current_y_values.flatten()

                #
                # modify modules with new setting values
                #
                for i in range(number_of_params):
                    target_module = pipeline.module(target_setting_module_list[i])
                    for setting in target_module.settings():
                        if setting.get_text() == target_setting_names_list[i]:
                            setting.set_value(new_target_settings[i])

                            #
                            # inform the pipeline about the edit
                            # pipeline re-runs from where the module has been changed; mind that pipeline-index is 1
                            # smaller than module number, so it needs to take module number of list -1
                            #
                            pipeline.edit_module(target_setting_module_list[i]-1, is_image_set_modification=False)

                #
                # ensure that CP is running the pipeline from the first module that was modified
                #
                pipeline.edit_module(min(target_setting_module_list) - 1, is_image_set_modification=False)

                #
                # if user wants to show the display-window, save data needed for display in workspace.display_data
                #
                if self.show_window:
                    #
                    # we first need to search for the lowest available y and the corresponding X settings
                    #
                    x = np.loadtxt(x_absolute_path)
                    y = np.loadtxt(y_absolute_path)

                    ind_best = np.argmin(y)

                    #
                    # account for case with 1D array in first round and several dimension issues
                    #
                    try:
                        num_dimensions = x.ndim
                    except AttributeError:
                        num_dimensions = 0

                    if num_dimensions == 1 and number_of_params == 1:  # otherwise would also reshape 1st round of x 2D
                        x = x.reshape(-1, 1)
                    elif num_dimensions == 0:  # accounts for x 1D in first round
                        x = np.array([x])
                    elif np.size(y) == 1:
                        x = np.array([x])

                    x_best = x[ind_best]
                    x_best = x_best.flatten()

                    workspace.display_data.statistics = []
                    for i in range(number_of_params):
                        workspace.display_data.statistics.append(
                            (target_setting_names_list[i], x_best[i], target_setting_values_list[i],
                             new_target_settings[i]))

                    workspace.display_data.col_labels = ("Setting Name", "Best Value so far", "Old Value", "New Value")
                    workspace.display_data.y_values = current_y_values

        #
        # no optimisation when quality is already satisfying
        #
        else:

            #
            # open or create x_file and write the final values of the setting parameters to it
            #
            with open(x_absolute_path, "a+") as x_file:
                for v in target_setting_values_list:
                    x_file.write("{} ".format(v))
                x_file.write("\n")

            info = "Quality satisfied. No Optimisation necessary."

            #
            # When quality is determined to be satisfying and an autoevaluation module exists, the final output of
            # the automated evaluation is displayed to the user and can be rated with satisfying or unsatisfying
            #

            if self.weighting_auto.value > 0:

                overlay_image = workspace.image_set.get_image("AutoEvaluationOverlay")
                #b = overlay_image.has_parent_image
                #print ("Has parent image: ")
                #print (b)

                base_image = overlay_image.parent_image

                base_image_pxdata = base_image.pixel_data

                #print ("Base image px data:")
                #print (base_image_pxdata)

                overlay_image_pxdata = overlay_image.pixel_data

                #print ("Overlay image px data:")
                #print (overlay_image_pxdata)

                #
                # Interrupt pipeline execution and send interaction request to workspace.
                # As the run-method is executed in a separate thread, it needs to give control to the UI thread.
                # The handle_interaction method will be called and gets the pixel data passed as parameters.
                # It will then return a result.
                #
                result = workspace.interaction_request(self, base_image_pxdata, overlay_image_pxdata)

                # if user finds the AutoEval result satisfying, do nothing and continue with pipeline run
                if result == 1:
                    #
                    # open or create y_file and write 0 to it as indicator that BO was not needed as quality is already
                    # satisfying or satisfying after some optimisation has already taken place
                    #
                    with open(y_absolute_path, "a+") as y_file:
                        y_file.write("{}\n".format(0))

                    print("OK button pressed, continuing pipeline run")

                # if user finds result unsatisfying, document this in y
                else:
                    #
                    # open or create y_file and write 1 to it as indicator that quality not satisfying
                    #
                    with open(y_absolute_path, "a+") as y_file:
                        y_file.write("{}\n".format(1))
                    print("Result not ok, documenting bad quality in y-value")
                    info = "Quality not satisfying. Please adjust ranges in AutoEvaluation module."

            else:
                #
                # open or create y_file and write 0 to it as indicator that BO was not needed as quality is already
                # satisfying or satisfying after some optimisation has already taken place
                #
                with open(y_absolute_path, "a+") as y_file:
                    y_file.write("{}\n".format(0))

            print("NO OPTIMISATION")

            if self.show_window:
                workspace.display_data.statistics = []
                for i in range(number_of_params):
                    workspace.display_data.statistics.append(
                        (target_setting_names_list[i], target_setting_values_list[i]))

                workspace.display_data.col_labels = ("Setting Name", "Final Best Value")

                workspace.display_data.stop_info = info

    #
    # if user wants to show the display window during pipeline execution, this method is called by UI thread
    # display the data saved in display_data of workspace
    # used a CP defined figure to plot/display data via matplotlib
    #
    def display(self, workspace, figure):
        #
        # data plotted when BO was run
        #
        if self.optimisation_on:
            #
            # create two subplots
            #
            figure.set_subplots((1, 2))

            #
            # prepare first plot showing a scatter plot with the development of y (quality indicator) over the rounds
            #
            num_y = np.size(workspace.display_data.y_values)

            x_values_axis = np.arange(1, num_y+1)

            figure.subplot_scatter(0, 0,
                                   x_values_axis, workspace.display_data.y_values,
                                   xlabel="Iteration", ylabel="Quality", title="Quality over iterations")

            #
            # prepare second plot showing a table with old and new values
            #
            figure.subplot_table(0, 1,
                                 workspace.display_data.statistics,
                                 col_labels=workspace.display_data.col_labels)

        #
        # information plotted when BO was not run or max_iter was reached
        #
        else:
            figure.set_subplots((1, 2))
            figure.set_subplot_title(workspace.display_data.stop_info, 0, 0)
            figure.subplot_table(0, 1,
                                 workspace.display_data.statistics,
                                 col_labels=workspace.display_data.col_labels)

    #
    # helper function:
    # Return a list of pipeline modules (only IdentifyObjects modules)
    #
    def get_module_list(self, pipeline):
        modules = pipeline.modules()
        module_list = []
        for module in modules:
            #
            # there is a filter for only using IdentifyObjects modules; the filter can be switchen on by uncommenting
            # the following two lines and commenting the 3rd line
            #
            # if "Identify" in module.module_name:
            #     module_list.append("{} #{}".format(module.module_name, module.get_module_num()))
            module_list.append("{} #{}".format(module.module_name, module.get_module_num()))
        return module_list

    #
    # helper function:
    # Return a list of settings from the chosen modules
    #
    def get_settings_from_modules(self, pipeline):
        setting_list = []
        modules = pipeline.modules()
        mod_name_list = []

        for parameter in self.parameters:
            name_list = parameter.module_names.value_text.split(" ")
            name = name_list[0]
            mod_name_list += [name]

        for module in modules:
            if module.module_name in mod_name_list:
                for setting in module.visible_settings():
                    setting_text = setting.get_text()
                    if setting_text not in setting_list:
                        setting_list.append(setting_text)

        return setting_list

    #
    # helper function:
    # Necessary to refresh the dropdown menus in GUI
    #
    def refreshGUI(self):
        print("GUI refreshed")

    #
    # helper function:
    # Deletes existing files storing previous values for x and y
    #
    def delete_data(self):
        #
        # create absolute pathname
        #
        x_filename = "x_bo_{}.txt".format(self.get_module_num())
        y_filename = "y_bo_{}.txt".format(self.get_module_num())

        x_absolute_path = "{}/{}".format(self.pathname.get_absolute_path(), x_filename)
        y_absolute_path = "{}/{}".format(self.pathname.get_absolute_path(), y_filename)

        #
        # remove files
        #
        os.remove(x_absolute_path)
        os.remove(y_absolute_path)

        print("Data deleted")

    ##############################################
    # Actual Bayesian optimisation functionality #
    ##############################################

    def bayesian_optimisation(self, manual_result, auto_evaulation_results,
                              values_list, setting_range, range_steps, num_params,
                              w_auto, w_manual, length_scale, alpha):

        #
        # need to load and write available data to files to persist it over the iterations; files contain x and y values
        # define the name of the files where x and y values are written to; names store the module number in case
        # BO module is used in more than one place of the pipeline
        #

        #
        # create absolute pathname
        #
        x_filename = "x_bo_{}.txt".format(self.get_module_num())
        y_filename = "y_bo_{}.txt".format(self.get_module_num())

        x_absolute_path = "{}/{}".format(self.pathname.get_absolute_path(), x_filename)
        y_absolute_path = "{}/{}".format(self.pathname.get_absolute_path(), y_filename)

        #
        # open or create x_file and write the values of the setting parameters to it
        #
        with open(x_absolute_path, "a+") as x_file:
            for v in values_list:
                x_file.write("{} ".format(v))
            x_file.write("\n")

        #
        # open or create y_file and write the values of the evaluation measurements to it
        # normalise y before writing it to the file
        #
        with open(y_absolute_path, "a+") as y_file:
            y_normalised = self.normalise_y(manual_result, auto_evaulation_results, w_manual, w_auto)
            y_file.write("{}\n".format(y_normalised))

        #
        # load the x and y values into numpy arrays
        # x values are the settings values
        # y values are the percentaged evaluation deviation values normalised and weighted to one single y value
        #
        x = np.loadtxt(x_absolute_path)
        y = np.loadtxt(y_absolute_path)

        #
        # Set up the actual iterative optimisation loop
        #
        n_offset_bayesopt = 2                           # min number of data points to start BO
        n_max_iter = int(self.max_iter.get_value())     # no. of max iterations
        n_current_iter = len(np.atleast_1d(y))          # number of data available

        #
        # set random generator;
        # use a flexible seed so that each round, different randomised numbers are chosen
        # this is necessary for the randomly chosen X when not enough data is yet available and
        # for the random first 10000 entries of the matrix to chose the candidate set from
        #
        np.random.seed(3*345 + n_current_iter)

        ########################################################################
        # create a suitable candidate set matrix based on the user input       #
        # take into account the range and steps a settings should be varied in #
        ########################################################################

        #
        # account for case with 1D array in first round and several dimension issues
        #
        try:
            num_dimensions = x.ndim
        except AttributeError:
            num_dimensions = 0

        if num_dimensions == 1 and num_params == 1:     # otherwise it would also reshape the first round of x 2D
            x = x.reshape(-1, 1)
        elif num_dimensions == 0:                       # accounts for x 1D in first round
            x = np.array([x])

        #
        # find out columns of x (num_cols)
        #
        try:
            num_cols = x.shape[1]
        except IndexError:
            num_cols = x.shape[0]

        #
        # create a 1D candidate set for each x dimension in the range and with the range steps given by user
        #
        candidate_arrays = []
        for i in range(0, num_cols):
            a = float(setting_range[i][0])
            b = float(setting_range[i][1])
            c = float(range_steps[i])

            candidate = np.arange(a, b, c)
            candidate_arrays += [candidate]

        # print("CANDIDATE ARRAYS")
        # print(candidate_arrays)

        #
        # create a matrix for either 1, 2, 3 or 4 parameters with all possible combinations of the 1D-arrays
        #
        if num_cols == 1:
            c = candidate_arrays[0]
        elif num_cols == 2:
            c = list(product(candidate_arrays[0], candidate_arrays[1]))
        elif num_cols == 3:
            c = list(product(candidate_arrays[0], candidate_arrays[1], candidate_arrays[2]))
        else:
            c = list(product(candidate_arrays[0], candidate_arrays[1], candidate_arrays[2], candidate_arrays[3]))

        unstandardised_candidates_array = np.asanyarray(c)

        #
        # if x is 1D, candidates array needs to be re-shaped
        #
        if num_cols == 1:
            unstandardised_candidates_array = unstandardised_candidates_array.reshape(-1, 1)

        #
        # initiate the correction of numbers in array:
        # standardisation of matrix entries; important step in Machine Learning
        # we need to calculate the mean and standard deviation of *all* candidates available (which includes the
        # already gathered X)
        # the numbers are used for later calculations
        #

        # 1st step: get mean from each column in matrix
        mean_candidates = np.mean(unstandardised_candidates_array, axis=0)

        # 2nd step: subtract mean from matrix
        cand_1 = unstandardised_candidates_array - mean_candidates

        # 3rd step: calculate standard deviation per column
        st_dev_candidates = np.std(cand_1, axis=0)

        #
        # check how many entries (rows) the candidate matrix has
        #
        num_entries = np.size(unstandardised_candidates_array, axis=0)
        # print("NUMBER OF ALL CANDIDATES")
        # print(num_entries)

        #
        # if candidate matrix is too large, take a subset of 10000 randomly chosen entries;
        # this ensures that the matrix has <= 10.000 row entries
        #
        if num_entries > 10000:
            np.take(unstandardised_candidates_array, np.random.permutation(unstandardised_candidates_array.shape[0]),
                    axis=0, out=unstandardised_candidates_array)
            unstandardised_candidates_array = unstandardised_candidates_array[:10000]

        #############################################
        # Init the data for the bayes opt procedure #
        #############################################

        #
        # we need to standardise the current set of x values with the calculated mean and standard deviation
        # we need to account for case with 1D array in first round
        #
        try:
            x.shape[1]
        except IndexError:
            x = np.array([x])

        x_1 = x - mean_candidates
        x_active_bayesopt = x_1 / st_dev_candidates

        # print("STANDARDISED X")
        # print(x_active_bayesopt)

        #
        # we need to remove the unstandardised x from unstandardised candidates and then standardise the remaining
        # candidates!
        # as comparing floating point numbers and arrays often results in faulty results, we need to transform
        # the floating point numbers into integers; the numbers used in CP do not have more than 3 decimals
        #

        #
        # we need to account for case with 1D array for unstandardised candidates
        #
        try:
            unstandardised_candidates_array.shape[1]
        except IndexError:
            unstandardised_candidates_array = np.array([unstandardised_candidates_array])
            unstandardised_candidates_array = unstandardised_candidates_array.reshape(-1, 1)

        #
        # transform numbers into integers by multiplying them with 1000
        #
        mul_std_cand = np.multiply(unstandardised_candidates_array, 1000)
        mul_std_cand = mul_std_cand.astype(int)
        mul_x = np.multiply(x, 1000)
        mul_x = mul_x.astype(int)

        #
        # transform arrays into 1D and use np.setdiff1d to extract all rows that are in the unstandardised candidates
        # but not in x
        #
        c_rows = mul_std_cand.view([('', mul_std_cand.dtype)] * mul_std_cand.shape[1])
        x_rows = mul_x.view([('', mul_x.dtype)] * mul_x.shape[1])

        new_mul_std_candidates = np.setdiff1d(c_rows, x_rows).view(
            mul_std_cand.dtype).reshape(-1, mul_std_cand.shape[1])

        #
        # transform the new array's numbers back into the original floating point numbers by dividing them by 1000
        #
        new_mul_std_candidates = new_mul_std_candidates.astype(float)
        new_candidates_bayesopt = np.divide(new_mul_std_candidates, 1000)   # these are unstandardised and without x

        #
        # now standardise remaining set with the calculated mean and standard deviation
        #
        cand_2 = new_candidates_bayesopt - mean_candidates
        candidates_bayesopt = cand_2 / st_dev_candidates

        # print("STANDARDISED CANDIDATES WITHOUT X")
        # print(candidates_bayesopt)

        #
        # account for the case of x being 1D
        # needs reshaping before fitting to the model
        #
        if num_cols == 1:
            x_active_bayesopt = x_active_bayesopt.reshape(-1, 1)
            candidates_bayesopt = candidates_bayesopt.reshape(-1, 1)

        #
        # check how many entries (rows) the matrix has now (testing only)
        #
        # num_entries1 = np.size(candidates_bayesopt, axis=0)
        # print("SIZE OF CANDIDATES WITHOUT X")
        # print(num_entries1)

        #
        # load the already available points y
        #
        y_active_bayesopt = y

        #
        # Run the procedure once and then return the new best x when no. of iterations is < than max_iter
        #
        if n_current_iter <= n_max_iter:
            print(" Iter: " + str(n_current_iter))

            #
            # Update Bayes opt active set with one point selected via EI
            # (of we have exceeded the initial offset period)
            #
            if n_current_iter > n_offset_bayesopt:

                ###################################
                # Bayesian Optimisation Procedure #
                ###################################

                print("EXECUTING BAYESIAN OPTIMISATION PROCEDURE")

                #
                # initialise the kernel (covariance function) for the BO model
                #
                kernel_init = gp.kernels.ConstantKernel(0.1) * gp.kernels.RBF(length_scale=length_scale)

                #
                # after 20 iterations there is enough data to use the optimizer to optimize the kernel's
                # hyperparameters
                #
                optimizer = None

                if n_current_iter >= 10:
                    optimizer = "fmin_l_bfgs_b"
                    # print("optimiser on")

                #
                # Define and fit the GP model (using the kernel_bayesopt_init parameters)
                #
                model_bayesopt = gp.GaussianProcessRegressor(kernel=deepcopy(kernel_init),
                                                             alpha=alpha,
                                                             n_restarts_optimizer=5,
                                                             optimizer=optimizer,
                                                             normalize_y=True)

                #
                # fit model with available active x and y parameters
                #
                model_bayesopt.fit(x_active_bayesopt, y_active_bayesopt)

                #
                # Find the currently best value (based on the model, not the active data itself as there could be
                # a tiny difference)
                #
                mu_active_bayesopt, sigma_active_bayesopt = model_bayesopt.predict(x_active_bayesopt,
                                                                                   return_std=True)
                ind_optimum = np.argmin(mu_active_bayesopt)
                mu_min_active_bayesopt = mu_active_bayesopt[ind_optimum]

                #
                # Predict the values for all the possible candidates in the candidate set using the fitted
                # model_bayesopt
                #
                mu_candidates, sigma_candidates = model_bayesopt.predict(candidates_bayesopt, return_std=True)

                #
                # Compute the expected improvement for all the candidates
                #
                z = (mu_min_active_bayesopt - mu_candidates) / sigma_candidates
                ei = (mu_min_active_bayesopt - mu_candidates) * norm.cdf(z) + sigma_candidates * norm.pdf(z)
                ei[sigma_candidates == 0.0] = 0.0   # Make sure to account for the case where sigma==0 to avoid
                # numerical issues (would be NaN otherwise)

                #
                # Find the candidate with the largest expected improvement and choose that one to query/include
                #
                eimax = np.max(ei)  # maximum expected improvement

                #
                # find all points with the same maximum value of ei in case there are more than one
                #
                i_eimax = np.argwhere(eimax == ei)
                i_max = np.random.randint(np.size(i_eimax, axis=0), size=1)  # choose randomly among them
                ind_new_candidate_as_index_in_cand_set = [i_eimax[i_max[0]]]

                #
                # get the new suggested x from the candidates
                #
                new_x_standardised = candidates_bayesopt[ind_new_candidate_as_index_in_cand_set]

            #
            # Skip bayes opt until we reach n_offset_bayesopt and select random points for inclusion
            # (sometimes it is a good idea to include a few random examples)
            #
            else:
                print("RANDOMLY choosing new X as not enough data is available")

                ii = np.random.randint(np.size(candidates_bayesopt, axis=0), size=1)
                new_x_standardised = candidates_bayesopt[ii]

            ###################
            # Return X values #
            ###################

            #
            # now that new setting values X form the candidate set were chosen, they first need to be converted back
            # by de-standardising them with the standard deviation and mean we have calculated earlier
            #
            next_x_meaned = new_x_standardised * st_dev_candidates
            next_x = next_x_meaned + mean_candidates

            #
            # return the X values to adjust the settings and getting a new y value from the user for next BO round;
            # round values to account for any floating point decimal inaccuracies caused earlier;
            # the CP settings usually only take integers or floating point numbers with up to 3 decimals
            #

            next_x_round = np.around(next_x, decimals=3)

            # print("NEXT X")
            # print(next_x_round)

            return next_x_round, y_active_bayesopt

        #
        # If the max number of iterations is reached, stop B.O.; indicating it with returning None instead of arrays
        #
        else:
            print("MAX ITERATIONS REACHED")
            return None, None

    #
    # helper function;
    # normalise the manual and auto evaluation results and return a weighted normalised value for y
    #
    def normalise_y(self, manual_result, auto_evaluation_results, w_manual, w_auto):

        if len(manual_result) == 0:
            auto = auto_evaluation_results

            result_accumulated = float(np.sum(auto)) / np.size(auto)

            result_norm = float(result_accumulated / 100)

        elif len(auto_evaluation_results) == 0:
            result_norm = float(manual_result) / 100

        else:
            #
            # both evaluation modules get a user defined weighting
            #
            manual = float(w_manual)/100 * manual_result
            auto = float(w_auto)/100 * auto_evaluation_results

            result_accumulated = float(np.sum(manual) + np.sum(auto) / np.size(auto))

            result_norm = float(result_accumulated / 100)

        return result_norm

    #
    # handle_interaction is called during the run of the pipeline when an interaction request was made;
    # control is passed to UI thread and user sees UI window created in the interaction method
    #
    def handle_interaction(self, base_pixel_data, out_pixel_data):
        #
        # import UI modules (WX and Matplotlib) to show a pop up window for user interaction
        #
        import wx
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_wxagg
        from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

        #
        # Make a wx.Dialog. "with" will garbage collect all of the UI resources when the user closes the dialog.
        #
        # This is how the dialog frame is structured:
        #
        # -------- WX Dialog frame ---------
        # |  ----- WX BoxSizer ----------  |
        # |  |  -- Matplotlib canvas -- |  |
        # |  |  |  ----- Toolbar ---- | |  |
        # |  |  |  ----- Figure ----- | |  |
        # |  |  |  |  --- Axes ---- | | |  |
        # |  |  |  |  |           | | | |  |
        # |  |  |  |  | AxesImage | | | |  |
        # |  |  |  |  |           | | | |  |
        # |  |  |  |  ------------- | | |  |
        # |  |  |  ------------------ | |  |
        # |  |  ----------------------- |  |
        # |  |  |-----WX BoxSizer-----| |  |
        # |  |  |  WX Rating Label    | |  |
        # |  |  |     + Buttons       | |  |
        # |  |  | ------------------- | |  |
        # |  |  ----------------------- |  |
        # |  ----------------------------  |
        # ----------------------------------
        #

        with wx.Dialog(None, title="Automated Evaluation finished.", size=(800, 650)) as dlg:

            self.message = ""

            #
            # A wx.Sizer automatically adjusts the size of a window's sub-windows
            #
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)

            #
            # The figure holds the plot and axes
            #
            figure = plt.figure()

            #
            # Define axes on the figure
            #
            axes = figure.add_axes((.05, .05, .9, .9))

            #
            # show the original image and overlays
            #
            axes.imshow(base_pixel_data, 'gray', interpolation='none')
            axes.imshow(out_pixel_data, 'gray', interpolation='none', alpha=0.5)

            #
            # canvas which renders the figure
            #
            canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(dlg, -1, figure)

            #
            # Create a default toolbar for the canvas
            #
            toolbar = NavigationToolbar(canvas)
            toolbar.Realize()

            #
            # Put the toolbar and the canvas in the dialog
            #
            dlg.Sizer.Add(toolbar, 0, wx.LEFT | wx.EXPAND)
            dlg.Sizer.Add(canvas, 1, wx.EXPAND)

            #
            # add horizontal Sizers to Frame Sizer
            #
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            dlg.Sizer.Add(hsizer, 2, wx.ALIGN_CENTER)

            hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
            dlg.Sizer.Add(hsizer2, 3, wx.ALIGN_CENTER)

            hsizer3 = wx.BoxSizer(wx.HORIZONTAL)
            dlg.Sizer.Add(hsizer3, 4, wx.ALIGN_CENTER)

            #
            # create info label and rating buttons
            #
            info_label = wx.StaticText(dlg, label="Automated Evaluation has finished.")

            info_label2 = wx.StaticText(dlg, label="If result is not satisfying, please press 'Quality not "
                                                   "satisfying', cancel pipeline execution manually "
                                                   "and adjust tolerance ranges in AutoEvaluation Module.")
            info_label2.Wrap(600)

            button_1 = wx.Button(dlg, size=(200, -1), label="Quality ok")
            button_2 = wx.Button(dlg, size=(200, -1), label="Quality not satisfying")

            #
            # add elements to horizontal sizer
            #
            hsizer.Add(info_label, 0, wx.ALIGN_CENTER)
            hsizer2.Add(info_label2, 0, wx.ALIGN_CENTER)
            hsizer3.Add(button_1, 0, wx.ALIGN_CENTER)
            hsizer3.Add(button_2, 0, wx.ALIGN_CENTER)

            #
            # "on_button" gets called when the button is pressed.
            #
            # button.Bind directs WX to handle a button press event
            # by calling "on_button" with the event.
            #
            # dlg.EndModal tells WX to close the dialog and return control
            # to the caller.
            #
            def on_button(event):
                b = event.GetEventObject().GetLabel()
                self.message = b
                dlg.EndModal(1)
                plt.close(figure)

            button_1.Bind(wx.EVT_BUTTON, on_button)
            button_2.Bind(wx.EVT_BUTTON, on_button)

            #
            # Layout and show the dialog
            #
            dlg.Layout()
            dlg.ShowModal()

            #
            # Return the quality measure set by button press (or window close; default = 0); if quality is
            # satisfying, return 1, if not return 0
            #
            if self.message == "Quality ok":
                return 1
            else:
                return 0
