# bayesopt4cellprofiler
*interactive machine learning approach for fast and robust cell profiling.

Related paper: https://www.biorxiv.org/content/10.1101/2020.02.20.956268v1.full


# Current version:
In order to use the developed plugins
- AutomatedEvaluation
- ManualEvaluation
- BayesianOptimisation

you will need to download a version of CellProfiler 3 from

http://cellprofiler.org/releases/

After installation, save the three plugins (python scrips) into a folder.
Go to the program's File > Preferences menu. Choose the folder as plugins source
and restart the program.

After the restart, the plugins should be available in the module list under the
"Advanced" category.

You can now drag and drop a pipeline file into the program as well as the
image sets. Example pipelines and image sets that utilise the developed plugins
for cell segmentation and focal adhesion segmentation are provided. 


# On the horison:
- Support for mixed-type parameters (e.g. discrete and continuous)
- Possibility to specify prior information about individual parameters to aid the optimisaiton.
- Support for relative judgements of quality (A/B evaluations)
- ... let us know if you have suggestions/ideas for new features or improvements.


# Questions:
For feature requests or questions, refer to the official [CP webpage](http://cellprofiler.org), 
[CP GitHub wiki](https://github.com/CellProfiler/CellProfiler/wiki) or create an issue in this repo.
