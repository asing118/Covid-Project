import pandas as pd
import numpy as np
import os
import datetime as dt


def load_data(file_name='.'):
    cur_dir = os.getcwd()
    os.chdir("../../data/input")
    raw = pd.read_csv(file_name)
    os.chdir(cur_dir)

    return raw


def save_data(data, file_name='.'):
    cur_dir = os.getcwd()
    os.chdir("../../data/output")
    data.to_csv(file_name, encoding='utf-8', index=False)
    os.chdir(cur_dir)


def nan_check(value):
    if np.isnan(value):
        return 0
    return value


def convert_to_julian(x):
    if x <= 1 or np.isnan(x):
        return 0
    return dt.datetime.fromordinal(int(x))


def weighted_static_data_compute(weights, data):
    return sum(weights.values * data.values)


def date_time_mean(data_frame):
    df = data_frame[(data_frame.values != 0)]
    return df.min() + (df - df.min()).mean()


def process_state_static_data(data_frame, state, isDateColumnIgnore):
    area_per_county = data_frame['CensusPopulation2010'].fillna(0) \
                      / data_frame['PopulationDensityperSqMile2010'].fillna(1).apply(lambda x: 1 if x == 0 else x)
    area_per_county_norm = area_per_county.div(area_per_county.sum())
    population_norm = data_frame['CensusPopulation2010'].fillna(0) \
                      / data_frame['CensusPopulation2010'].apply(lambda x: 1 if x == 0 else x).sum()

    columns_to_be_area_weighted = [
        'PopulationDensityperSqMile2010',
    ]

    columns_to_be_population_weighted = [
        'MedianAge2010',
        'HeartDiseaseMortality',
        'StrokeMortality',
        'Smokers_Percentage',
        'RespMortalityRate2014',
        'SVIPercentile',
        'HPSAServedPop',
        'HPSAUnderservedPop',
        '3-YrDiabetes2015-17'
    ]

    date_columns = [
        'stay at home',
        '>50 gatherings',
        '>500 gatherings',
        'public schools',
        'restaurant dine-in',
        'entertainment/gym',
        'federal guidelines',
        'foreign travel ban'
    ]

    mask_data = [
        'forMask',
        'notForMask'
    ]

    result = []
    for column in data_frame.columns.values:
        if column == 'state':
            result.append(state)
        elif column in columns_to_be_area_weighted:
            result.append(weighted_static_data_compute(area_per_county_norm, data_frame[column].fillna(0)))
        elif column in columns_to_be_population_weighted:
            result.append(weighted_static_data_compute(population_norm, data_frame[column].fillna(0)))
        elif column in date_columns:
            if isDateColumnIgnore:
                result.append(date_time_mean(data_frame.dropna(subset=[column])[column]))
            else:
                result.append(convert_to_julian(data_frame.dropna(subset=[column])[column].mean()))
        elif column in mask_data:
            result.append(data_frame[column].mean())
        else:
            result.append(data_frame[column].fillna(0).sum())

    return result


def get_us_state_abbrev():
    us_state_abbrev = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AS': 'American Samoa', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
        'FL': 'Florida', 'GA': 'Georgia', 'GU': 'Guam', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
        'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
        'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
        'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
        'MP': 'Northern Mariana Islands', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
        'PR': 'Puerto Rico', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
        'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VI': 'Virgin Islands', 'VA': 'Virginia', 'WA': 'Washington',
        'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }

    return dict([(value, key) for key, value in us_state_abbrev.items()])


def process_google_mobility_data():

    # Data Source: https://www.google.com/covid19/mobility/

    columns_to_select = [
        'sub_region_1',
        'date',
        'retail_and_recreation_percent_change_from_baseline',
        'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline',
        'residential_percent_change_from_baseline'
    ]

    raw_google_mobility_data = load_data('GOOGLE_MOBILITY_STATE_DATE_WISE.csv')
    raw_google_mobility_data = raw_google_mobility_data[columns_to_select]
    raw_google_mobility_data[columns_to_select[2:]] = raw_google_mobility_data[columns_to_select[2:]].div(100).round(2)
    raw_google_mobility_data['date'] = pd.to_datetime(raw_google_mobility_data['date'].str.strip(), format='%m/%d/%Y')\
        .dt.strftime('%m-%d-%Y')
    raw_google_mobility_data['sub_region_1'].replace(get_us_state_abbrev(), inplace=True)
    raw_google_mobility_data = raw_google_mobility_data.groupby(['sub_region_1', 'date'], as_index=False).mean()
    raw_google_mobility_data.rename(columns={'sub_region_1': 'state'}, inplace=True)
    print('Done processing google mobility data')

    return raw_google_mobility_data


def process_infection_data():

    # Data source: New York Times

    columns_to_select = [
        'date',
        'state',
        'cases',
        'deaths'
    ]
    raw_infection_data = load_data('NYT_us-states.csv')
    raw_infection_data = raw_infection_data[columns_to_select]
    raw_infection_data['date'] = pd.to_datetime(raw_infection_data['date'].str.strip(), format='%m/%d/%y') \
        .dt.strftime('%m-%d-%Y')
    raw_infection_data['state'].replace(get_us_state_abbrev(), inplace=True)

    prev_state = ''
    prev_infection = 0
    prev_death = 0
    for i, row in raw_infection_data.iterrows():
        if row['state'] == prev_state:
            temp_prev_infection = row['cases']
            temp_prev_death = row['deaths']
            raw_infection_data.at[i, 'cases'] = row['cases'] - prev_infection
            raw_infection_data.at[i, 'deaths'] = row['deaths'] - prev_death
            prev_infection = temp_prev_infection
            prev_death = temp_prev_death
        else:
            prev_state = row['state']
            prev_infection = row['cases']
            prev_death = row['deaths']

    print('Done processing NYT infection data')

    return raw_infection_data


def process_apple_mobility_data():

    # Data source: https://www.apple.com/covid19/mobility
    raw_apple_mobility_data = load_data('Apple_mobility.csv')
    raw_apple_mobility_data['sub-region'].replace(get_us_state_abbrev(), inplace=True)
    date_columns = raw_apple_mobility_data.columns.values[6:]

    processed_data = []
    city_weight = 10
    county_weight = 1
    for key, state in get_us_state_abbrev().items():
        state_data = raw_apple_mobility_data.loc[raw_apple_mobility_data['sub-region'] == state]
        for date in date_columns:
            state_date_data = state_data[['geo_type', 'transportation_type', date]]
            state_date_data[[date]] = state_date_data[[date]].apply(lambda x: x - 100).div(100).round(2)
            city_walking = state_date_data.loc[np.logical_and(state_date_data['geo_type'] == 'city',
                                                              state_date_data['transportation_type'] == 'walking')]
            city_transit = state_date_data.loc[np.logical_and(state_date_data['geo_type'] == 'city',
                                                              state_date_data['transportation_type'] == 'transit')]
            city_driving = state_date_data.loc[np.logical_and(state_date_data['geo_type'] == 'city',
                                                              state_date_data['transportation_type'] == 'driving')]
            county_walking = state_date_data.loc[np.logical_and(state_date_data['geo_type'] == 'county',
                                                                state_date_data['transportation_type'] == 'walking')]
            county_transit = state_date_data.loc[np.logical_and(state_date_data['geo_type'] == 'county',
                                                                state_date_data['transportation_type'] == 'transit')]
            county_driving = state_date_data.loc[np.logical_and(state_date_data['geo_type'] == 'county',
                                                                state_date_data['transportation_type'] == 'driving')]
            processed_data.append([
                state,
                date,
                (nan_check(city_walking[date].mean()) * city_weight +
                 nan_check(county_walking[date].mean()) * county_weight) /
                (city_weight + county_weight),
                (nan_check(city_transit[date].mean()) * city_weight +
                 nan_check(county_transit[date].mean()) * county_weight) /
                (city_weight + county_weight),
                (nan_check(city_driving[date].mean()) * city_weight +
                 nan_check(county_driving[date].mean()) * county_weight) /
                (city_weight + county_weight)
            ])

    processed_df = pd.DataFrame(processed_data, columns=['state', 'date', 'walking', 'transit', 'driving'])
    processed_df['date'] = pd.to_datetime(processed_df['date'].str.strip(), format='%m/%d/%y') \
        .dt.strftime('%m-%d-%Y')

    print('Done processing apple mobility data')

    return processed_df


def process_us_data_dynamic(data_frame):
    data_frame = data_frame.drop('state', 1)

    data_frame = data_frame.groupby(['date'], as_index=False).agg({
            'retail_and_recreation_percent_change_from_baseline': 'mean',
            'grocery_and_pharmacy_percent_change_from_baseline': 'mean',
            'parks_percent_change_from_baseline': 'mean',
            'transit_stations_percent_change_from_baseline': 'mean',
            'workplaces_percent_change_from_baseline': 'mean',
            'residential_percent_change_from_baseline': 'mean',
            'cases': 'sum',
            'deaths': 'sum'
        })
    return data_frame


def generate_processed_dynamic_data():
    # Dynamic data by date processing

    google_mobility_df = process_google_mobility_data()
    infection_data_df = process_infection_data()
    apple_mobility_df = process_apple_mobility_data()

    overall_df = pd.merge(google_mobility_df, apple_mobility_df, how='outer', on=['state', 'date']).fillna(0)
    overall_df = pd.merge(overall_df, infection_data_df, how='outer', on=['state', 'date']).fillna(0)
    overall_df.sort_values(['state', 'date'], inplace=True)
    save_data(overall_df, 'processed_dynamic_data_by_state.csv')

    # get overall data for US
    process_us_data_df = process_us_data_dynamic(overall_df)
    process_us_data_df.sort_values(['date'], inplace=True)
    save_data(process_us_data_df, 'processed_dynamic_data_US.csv')

    print('Generated processed dynamic data')


def process_face_mask_data():

    # Data source:
    # https://today.yougov.com/topics/politics/articles-reports/2020/05/08/states-are-more-and-less-likely-adopt-face-masks
    raw_face_mask_data = load_data('Face_Mask_Survey.csv')
    raw_face_mask_data['state'].replace(get_us_state_abbrev(), inplace=True)
    raw_face_mask_data[['forMask']] = raw_face_mask_data[['forMask']].div(100).round(2)
    raw_face_mask_data[['notForMask']] = raw_face_mask_data[['notForMask']].div(100).round(2)

    print('Done processing face mask data')

    return raw_face_mask_data


def process_county_data():

    # Data source:
    # https://github.com/Yu-Group/covid19-severity-prediction/blob/master/data/county_data_abridged.csv
    raw_county_data = load_data('county_data_abridged.csv')

    processed_static_data_by_state = []
    for key, state in get_us_state_abbrev().items():
        state_data = raw_county_data.loc[raw_county_data['state'] == state]
        processed_static_data_by_state.append(process_state_static_data(state_data, state, False))

    processed_static_data_by_state_df = pd.DataFrame(processed_static_data_by_state,
                                                     columns=raw_county_data.columns.values)

    print('Done processing county data')

    return processed_static_data_by_state_df


def process_us_data_static(data_frame):
    processed_data = [process_state_static_data(data_frame, 'USA', True)]

    return pd.DataFrame(processed_data, columns=data_frame.columns.values)


def generate_processed_static_data():
    face_mask_df = process_face_mask_data()
    county_data_df = process_county_data()

    overall_df = pd.merge(county_data_df, face_mask_df, how='outer', on=['state']).fillna(0)
    overall_df['notForMask'] = overall_df['notForMask'].apply(lambda x: 1 if x == 0 else x)
    save_data(overall_df, 'processed_static_data_by_state.csv')

    processed_us_data_df = process_us_data_static(overall_df);
    save_data(processed_us_data_df, 'processed_static_data_US.csv')

    print('Generated static data')


if __name__ == '__main__':

    generate_processed_dynamic_data()
    generate_processed_static_data()

