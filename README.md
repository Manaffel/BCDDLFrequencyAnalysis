# BCDDLFrequencyAnalysis
Tool to analyse BCDDL data \\\
BCDDLFrequencyAnalysis.py: \\\
This tool can be used to analyse the measured BCDDL respone to a given input signal.
The coherence can be analysed to choose an appropriate number of subsets, the Frequency 
Response Function (FRF) can be visualised via bode plots, a second order low pass 
with delay can be fitted to the data, and the delay of a high order TF can be fitted to the data. \\\
getTF.m: \\\
Fits a high order TF to the measured FRF. The derived numerator and denominator parameters can be used in BCDDLFrequencyAnalysis.py to fit the delay.
