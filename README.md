# EV_growth
A study on the consumption of copper to scale with the growth of electric vehicle implementation by 2030


# Title
Cu on the Otherside

## Readme Outline
- [This project aims at ]
    - [Scenario]
    - [Goals](#goals)
        - [Deliverables](#deliverables)
    - [Project Dependencies](#dependencies)

- [About the data](#data)
    - Scope
    - Acquiring
    - Preparing
    - Data Dictionary
    - Key Findings

- [Project Planning](#plan)  
    - Hypothesis
    - Steps
    - Conclusion



# About the project <a name="project_desc"></a>

## Scenario

The transition to full electic vehicles is well underway in the United States. Climate change is at the forefront of politics. President Biden proposed a climate change initiative aiming for 50% of all vehicles sold in the United States to be electric by 2030. In addition to 50% electric vehicle sales, President Biden also propsed 500,000 additional chargers to accomodate the amount of new EVs on the market. An electric vehicle uses almost four times the amount of copper than a regular car at almost 200 pounds per passenger car, while commericals vehicles can use up to 800 pounds. Our goal is to use time forecasting to predict the total amount of new us vehicle sales sold in 2030, total refined copper produced both in the US and world, to see if this the proposal is feasible. 



## Goals

- First we will use machine learning and time forecasting to predict total New US car sales in 2030. Using the predicted forecast, we will calcuate the proposed 50% sale goal and use that goal to determine the amount of copper needed to meet that goal.
- Second we again use machine learning and time forecasting to predict the yearly total mined refined copper produced both domestically and globally. Utilizing the forecast and the current rate of copper allocated for transportation of 16% to determine feasibility of the proposal.
- Additionally, we extrapolated the 500,000 proposed charging stations out to 2030 to assess the needed copper to meet the demands of the proposals.  

### Deliverables

- Showcase the results of our investivations in a presentation delivered to stakeholders.
- Create a reproducible jupyter notebook report that includes process, takeaways, and discoveries from every stage of the multi-pipelines
- A whitepaper to showcase the research and findings from our investigations. 
- Please find the deliverables here:

## Reproducing this project

- ** We have included both the cleaned and prepped data CSVs as well as the unprepped. For easier reproducability, please use the cleaned and prepped data.
1. Read and follow this README.md
2. Download the prepped or raw dataset and the .py files listed below:
    -
    -
    -
3. Finally run our ***FINAL NOTEBOOK** replicate our models and analysis

### Dependencies

This project makes use of several technologies that will need to be installed
* [![python-shield](https://img.shields.io/badge/Python-3-blue?&logo=python&logoColor=white)
    ](https://www.python.org/)
* [![jupyter-shield](https://img.shields.io/badge/Jupyter-notebook-orange?logo=jupyter&logoColor=white)
    ](https://jupyter.org/)
* [![numpy-shield](https://img.shields.io/badge/Numpy-grey?&logo=numpy)
    ](https://numpy.org/)
* [![pandas-shield](https://img.shields.io/badge/Pandas-grey?&logo=pandas)
    ](https://pandas.pydata.org/)
* [![matplotlib-shield](https://img.shields.io/badge/Matplotlib-grey.svg?)
    ](https://matplotlib.org)
* [![seaborn-shield](https://img.shields.io/badge/Seaborn-grey?&logoColor=white)
    ](https://seaborn.pydata.org/)
* [![scipy-shield](https://img.shields.io/badge/SciPy-grey?&logo=scipy&logoColor=white)
    ](https://scipy.org/)
* [![sklearn-shield](https://img.shields.io/badge/_-grey?logo=scikitlearn&logoColor=white&label=scikit-learn)
    ](https://scikit-learn.org/stable/)

Dependencies can be installed quickly with just a few lines of code.
```
%pip install notebook
%pip install numpy
%pip install pandas
%pip install matplotlib
%pip install seaborn
%pip install scipy
%pip install sklearn
```


# About the data

- Total New Car Sales: https://www.bts.gov/content/annual-us-motor-vehicle-production-and-factory-wholesale-sales-thousands-units
- US Copper mining production/consumption: https://www.usgs.gov/centers/national-minerals-information-center/copper-statistics-and-information
    - To extract the data from the PDF we used Tabula
- Historic Prices of Copper - https://www.macrotrends.net/1476/copper-prices-historical-chart-data
## Scope

{ How many records/columns? How many nulls? Does this project focus on a particular subset of the overall data? }

## Acquiring

- The data was acquired from the "About the Data" section


## Preparing

- Each dataset was prepared in individual notebooks. Please see the perspective notebook on how hte data was prepared. 
- If the raw data is used: follow the functions in prespective notebooks to have the data split and ready to explore for analysis.

## Exploring

- Ran vizulations to determine seasonality, trands, and how each target variable changed over time. 
- Ran statistical tests to determine stationality of the data and the correlation of lag
- Data was processed during the prep stage for exploration

## Modeling/Evaluation

- Developed a baseline by taking a simple moving average for each target variable
- Due to the small amount of available data, we split the data only into train and test. This gave us the most amount of data to train the models on. 
- Built multiple forecasting models to improve upon baseline accuracy. 

## Deliverables

- A final reproducable jupyter notebook
- Slideshow to show the important findings during the investigation
- Whitepaper to detail findings and projections in more detail


## Data Dictionary

# Project Planning <a name="plan"></a>

- 

## Initial Hypotheses



__Sources:__

[Historic Prices of Copper] - https://www.macrotrends.net/1476/copper-prices-historical-chart-data - historic_prices.csv

[Domestic Auto Production] - https://fred.stlouisfed.org/series/DAUPSA - auto_production.csv