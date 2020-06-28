import click
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_name='.'):
    cur_dir = os.getcwd()
    os.chdir("data/output")
    raw = pd.read_csv(file_name)
    os.chdir(cur_dir)

    return raw


def get_mobility_data_mapping_dict():
    mapping_dict = {
        'date': 'date',
        'state': 'state',
        'retail_and_recreation_percent_change_from_baseline': 'retail_and_recreation_change_fraction',
        'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_and_pharmacy_change_fraction',
        'parks_percent_change_from_baseline': 'parks_change_fraction',
        'transit_stations_percent_change_from_baseline': 'transit_stations_change_fraction',
        'workplaces_percent_change_from_baseline': 'workplaces_change_fraction',
        'residential_percent_change_from_baseline': 'residential_change_fraction',
        'walking': 'walking_change_fraction',
        'transit': 'driving_change_fraction',
        'driving': 'transit_change_fraction',
        'cases': 'cases',
        'deaths': 'deaths'
    }
    return mapping_dict


def show_data_for_state_date(state, date):
    try:
        date = datetime.strptime(date, '%m-%d-%y').strftime('%m-%d-%Y')
    except ValueError:
        print('Date time not in required format of mm-dd-yy')

    processed_data = load_data('processed_dynamic_data_by_state.csv')
    filtered_data = processed_data.loc[(processed_data['state'] == state) & (processed_data['date'] == date)]\
        .to_dict(orient='records')

    try:
        for key, value in filtered_data[0].items():
            print(get_mobility_data_mapping_dict()[str(key)] + " : " + str(value))
    except IndexError:
        print("No mobility data found for state: " + state + " and date: " + date)


def plot_cases_trend_graph(state):
    processed_data = load_data('processed_dynamic_data_by_state.csv')
    filtered_data = processed_data.loc[(processed_data['state'] == state)][['date', 'cases', 'deaths']]

    if filtered_data.shape[0] == 0:
        print("No data found for state: " + state)
        return

    date = filtered_data['date'].values
    cases = filtered_data['cases'].values
    deaths = filtered_data['deaths'].values

    plt.plot(date, cases)
    plt.plot(date, deaths)
    plt.xticks(np.arange(0, len(date), 20), rotation=60)
    plt.legend(['cases', 'deaths'], loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_mobility_trend(state, mobility_param):
    mapping_dict = get_mobility_data_mapping_dict()
    mapping_dict_rev = dict([(value, key) for key, value in mapping_dict.items()])

    try:
        mapping_dict_rev[mobility_param]
    except KeyError:
        print("mobility parameter not valid. Please check help for accepted set of parameters")
        return

    processed_data = load_data('processed_dynamic_data_by_state.csv')
    date = processed_data.loc[(processed_data['state'] == state)]['date']
    mobility_data = processed_data.loc[(processed_data['state'] == state)][mapping_dict_rev[mobility_param]]

    if mobility_data.shape[0] == 0:
        print("No data found for state: " + state)
        return

    plt.plot(date, mobility_data)
    plt.xticks(np.arange(0, len(date), 20), rotation=60)
    plt.legend([mobility_param], loc='upper left')
    plt.tight_layout()
    plt.show()


@click.command()
@click.option('--operation', default='show_mobility_data_on_date',
              help='Valid operations: show_mobility_data_on_date, get_prediction, show_cases_trend, '
                   'show_mobility_trend')
@click.option('--state', default='AZ', help='State code')
@click.option('--date', default='01-13-20', help='Date in mm-dd-yy format')
@click.option('--retail_and_recreation_change_fraction', default=-2.0,
              help="retail and recreation change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--grocery_and_pharmacy_change_fraction', default=-2.0,
              help="grocery and pharmacy change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--parks_change_fraction', default=-2.0,
              help="parks change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--transit_stations_change_fraction', default=-2.0,
              help="transit stations change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--workplaces_change_fraction', default=-2.0,
              help="workplaces change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--residential_change_fraction', default=-2.0,
              help="residential change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--walking_change_fraction', default=-2.0,
              help="walking change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--driving_change_fraction', default=-2.0,
              help="driving change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--transit_change_fraction', default=-2.0,
              help="transit change fraction. baseline is 0 and value should be between -1 to 1")
@click.option('--mobility_param', default='retail_and_recreation_change_fraction',
              help="one of the mobility parameters. accepted set of options are mobility param values from help")
def hello(operation,
          state,
          date,
          retail_and_recreation_change_fraction,
          grocery_and_pharmacy_change_fraction,
          parks_change_fraction,
          transit_stations_change_fraction,
          workplaces_change_fraction,
          residential_change_fraction,
          walking_change_fraction,
          driving_change_fraction,
          transit_change_fraction,
          mobility_param
          ):
    """Simple program that greets NAME for a total of COUNT times."""
    if operation == 'show_mobility_data_on_date':
        show_data_for_state_date(state, date)
    elif operation == 'show_cases_trend':
        plot_cases_trend_graph(state)
    elif operation == 'show_mobility_trend':
        plot_mobility_trend(state, mobility_param)


if __name__ == '__main__':
    hello()
