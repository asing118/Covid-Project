# Covid-Project

## Overview
This project entails prediction of new Covid cases for different states based on users mobility data per day and different state wise static data like 
government restrictions, demographic informations etc. The project can be pulled and a CLI based tool is supported for data visualization and prediction analysis.


## CLI based tool
### Requirements
- Must have python3 and location of interpreter set in python path variable. Use pip3 to install library functions not in your system.
- Run the cli command from root of the project

### Data visualization
#### Get mobility data for a particular state on a paricular day

```
python3 src/CommandLineInterface.py --operation show_mobility_data_on_date --state CA --date 03-13-2020
```

```
--operation Operation name to be performed. By default it is set to show_mobility_data_on_date
--state the state code for which data is to be displayed. By default is set to AZ
--date the date in mm-dd-yy format for which the data is to be displayed. Default is 01-13-2020
```
#### Generate trend graph for cases and deaths per day for a given state

```
python3 src/CommandLineInterface.py --operation show_cases_trend --state CA
```

#### Get mobility data trend graph for a particular mobility parameter of a particular state

```
python3 src/CommandLineInterface.py --operation show_mobility_trend --state CA --mobility_param grocery_and_pharmacy_change_fraction
```

```
--mobility_param the mobility parameter for which the action is to be performed. Default is retail_and_recreation_change_fraction

Allowed values for mobility_param:
- retail_and_recreation_change_fraction
- grocery_and_pharmacy_change_fraction
- parks_change_fraction
- transit_stations_change_fraction
- workplaces_change_fraction
- residential_change_fraction
- walking_change_fraction
- driving_change_fraction
- transit_change_fraction
```

### Data prediction
Based on user defined input parameter for one of the mobility param for last 8 days for a give state, a prediction output for the state containing 
next 14 days prediction of cases on original mobility param values, next 14 days prediction of cases on user defined mobility param values and original next 14 days
number of cases is displayed. This uses a pretrained model in LSTM. Currently supported states are NY and AZ and will be extended in future.

```
python3 src/CommandLineInterface.py --operation get_prediction --state CA --mobility_param grocery_and_pharmacy_change_fraction --date 03-13-2020 --param_values '[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]'
```

```
--param_values user defined mobility param values for a particular mobility param for last 8 days. Default is '[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]'
--date start date from which the mobility param is altered.
```
