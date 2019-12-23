pyNLPAR
========

Python implementation of the Non-Local Pattern Averaging Reindexing (NLPAR) method for EBSD Kikuchi pattern denoising.
The method was described in Brewick et al., 2019, "NLPAR: Non-local smoothing for enhanced EBSD pattern indexing" (see DOI: https://doi.org/10.1016/j.ultramic.2019.02.013)
It consists of a non local denoising of Kikuchi pattern by averaging a given pattern with his neighbors using a weighted sum. 
The weights are computed thanks to a similarity metric acounting for average noise level an similarity with the neighbours.

Motivations
============

The original article implemented the NLPAR algorithm using IDL programming langage which requires a commercial licence to be used.
The motivation of the present code is to implement the algorithm described in the source using python as it is a free langage, maybe at the cost of computation efficiency