# Multi Commodity Network Flow

Using Dantzig-Wolfe decomposition algorithm to solve MCNF(Multi Commodity Network Flow) problem. Final project for the course of (large-scale) linear programming.

## Background

We know the structure of urban road network and people's travel demand. There is limited capacity on each road segment. If the influence of the traffic flow on the road segment on the road travel time (i.e., the congestion effect) is not considered. Assume that all vehicles are dispatched by a central decision maker (i.e., no game exists). Then how to arrange the route selection between each pair of origin-destination (OD) and the traffic flow on each route so that the sum of travel costs between all OD pairs is minimized?

## Program

Required packages: numpy, pandas, **gurobipy**

## Explanation of the Algorithm

See [here](0106_Opt_Final.pdf) (in Chinese). Comment on code in English.