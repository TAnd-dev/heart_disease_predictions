import pandas as pd
import numpy as np


def create_general_analysis(df, asc=False):
    unique_values = [df[i].sort_values(ascending=asc).unique() for i in df]
    table_info = pd.DataFrame(
        {
            'values_num': df.count(),
            'nan_values_num': df.isna().sum(),
            'occupancy': (1 - df.isna().sum() / df.count()) * 100,
            'unique_values_num': df.nunique(),
            'min_value': df.min(),
            'max_value': df.max(),
            'unique_values': unique_values,
            'dtype': df.dtypes
        }
    )

    print('General data analysis: \n')
    print('Shape of the Data Frame: ', df.shape)
    print(
        f'Duplicates in the Data Frame: {df.duplicated().sum()}, ({round(df.duplicated().sum() / df.shape[0], 4) * 100})')

    return table_info


def is_monotonic(temp_series):
    return (
            all(temp_series[i] >= temp_series[i + 1] for i in range(len(temp_series) - 1))
            or
            all(temp_series[i] <= temp_series[i + 1] for i in range(len(temp_series) - 1))
    )


def prepare_bins(bin_data, col, target_col, max_bins):
    is_monotonic_bin = False
    is_binned = False
    remark = None

    for n_bins in range(max_bins, 2, -1):
        bin_data[f'{col}_bins'] = pd.qcut(bin_data[col], n_bins, duplicates='drop')
        monotonic_series = bin_data.groupby(f'{col}_bins')[target_col].mean().reset_index(drop=True)

        if is_monotonic(monotonic_series):
            is_monotonic_bin = True
            is_binned = True
            remark = 'binned monotonically'
            break

    if not is_monotonic_bin:
        min_val = bin_data[col].min()
        mean_val = bin_data[col].mean()
        max_val = bin_data[col].max()
        bin_data[f'{col}_bins'] = pd.cut(bin_data[col], [min_val, mean_val, max_val], include_lowest=True)

    if bin_data[f'{col}_bins'].nunique() == 2:
        is_binned = True
        remark = 'binned forcefully'

    if is_binned:
        class_col = f'{col}_bins'
        binned_data = bin_data[[col, class_col, target_col]].copy()
    else:
        remark = "couldn't bin"
        class_col = col
        binned_data = bin_data[[class_col, target_col]].copy()

    return class_col, remark, binned_data


def get_iv_woe_bin(binned_data, target_col, class_col):
    if '_bins' in class_col:
        binned_data[class_col] = binned_data[class_col].cat.add_categories('Missing')
        binned_data[class_col] = binned_data[class_col].fillna('Missing')
        temp_groupby = (
            binned_data
            .groupby(
                class_col, as_index=False
            )
            .agg(
                {
                    class_col.replace('_bins', ''): ['min', 'max'],
                    target_col: ['count', 'sum', 'mean']
                }
            )
        )
    else:
        binned_data[class_col] = binned_data[class_col].fillna('Missing')
        temp_groupby = (
            binned_data
            .groupby(class_col)
            .agg(
                {
                    class_col: ['first', 'first'],
                    target_col: ['count', 'sum', 'mean']
                }
            ).reset_index()
        )
    temp_groupby.columns = ['sample_class', 'min_value', 'max_value', 'sample_count', 'event_count', 'event_rate']
    temp_groupby['non_event_count'] = temp_groupby['sample_count'] - temp_groupby['event_count']
    temp_groupby['non_event_rate'] = 1 - temp_groupby['event_rate']

    if '_bins' not in class_col and ('Missing' in temp_groupby['min_value'] or 'Missing' in temp_groupby['max_value']):
        temp_groupby['min_value'] = temp_groupby['min_value'].replace({'Missing': np.nan})
        temp_groupby['max_value'] = temp_groupby['max_value'].replace({'Missing': np.nan})

    temp_groupby['feature'] = class_col

    if '_bins' in class_col:
        temp_groupby['sample_class_label'] = temp_groupby['sample_class'].replace({'Missing': np.nan}).astype(
            'category').cat.codes.replace({-1: np.nan})
    else:
        temp_groupby['sample_class_label'] = np.nan

    temp_groupby['distbn_non_event'] = temp_groupby['non_event_count'] / temp_groupby['non_event_count'].sum()
    temp_groupby['distbn_event'] = temp_groupby['event_count'] / temp_groupby['event_count'].sum()

    temp_groupby['woe'] = np.log(temp_groupby['distbn_non_event'] / temp_groupby['distbn_event'])
    temp_groupby['iv'] = (temp_groupby['distbn_non_event'] - temp_groupby['distbn_event']) * temp_groupby['woe']

    temp_groupby['woe'] = temp_groupby['woe'].replace([np.inf, -np.inf], 0)
    temp_groupby['iv'] = temp_groupby['iv'].replace([np.inf, -np.inf], 0)

    return temp_groupby


def get_iv_woe(data, target_col, max_bins):
    iv_woe = pd.DataFrame()
    remarks_list = []

    for col in data.columns:
        if col == target_col:
            continue
        if np.issubdtype(data[col], np.number) and data[col].nunique() > 2:
            class_col, remark, binned_data = prepare_bins(data[[col, target_col]].copy(), col, target_col, max_bins)
            agg_data = get_iv_woe_bin(binned_data.copy(), target_col, class_col)
            remarks_list.append({'feature': col, 'remark': remark})
        else:
            agg_data = get_iv_woe_bin(data[[col, target_col]].copy(), target_col, col)
            remarks_list.append({'feature': col, 'remark': 'categorical'})

        iv_woe = pd.concat([iv_woe, agg_data])

    remarks_list = pd.DataFrame(remarks_list)

    iv_woe['feature'] = iv_woe['feature'].replace('_bins', '', regex=True)
    iv_woe = iv_woe[["feature", "sample_class", "sample_class_label", "sample_count", "min_value", "max_value",
                     "non_event_count", "non_event_rate", "event_count", "event_rate", 'distbn_non_event',
                     'distbn_event', 'woe', 'iv']]

    iv = iv_woe.groupby('feature', as_index=False)['iv'].agg(['sum', 'count'])
    iv.columns = ['feature', 'iv', 'number_of_classes']

    null_percent_data = data.isnull().mean().reset_index()
    null_percent_data.columns = ['feature', 'feature_null_percent']

    iv = iv.merge(null_percent_data, on='feature', how='left')
    iv = iv.merge(remarks_list, on='feature', how='left')

    iv_woe = iv_woe.merge(iv[['feature', 'iv', 'remark']].rename(columns={'iv': 'iv_sum'}), on='feature', how='left')

    return iv, iv_woe.replace({'Missing': np.nan})


def get_oldpeak_cat(val):
    if val < 0:
        return '0-'
    elif val == 0:
        return '[0-0]'
    elif 0 < val <= 1:
        return '(0 - 1]'
    elif 1 < val <= 2:
        return '(1 - 2]'
    elif 2 < val <= 3:
        return '(2 - 3]'
    return '3+'


def get_age_cat(age):
    if age <= 32.9:
        return '(27.951, 32.9]'
    elif age <= 37.8:
        return '(32.9, 37.8]'
    elif age <= 42.7:
        return '(37.8, 42.7]'
    elif age <= 47.6:
        return '(42.7, 47.6]'
    elif age <= 52.5:
        return '(47.6, 52.5]'
    elif age <= 57.4:
        return '(52.5, 57.4]'
    elif age <= 62.3:
        return '(57.4, 62.3]'
    elif age <= 67.2:
        return '(62.3, 67.2]'
    elif age <= 72.1:
        return '(67.2, 72.1]'
    return '(72.1, 77.0]'


def get_resting_bp_cat(resting_bp):
    if resting_bp <= 120:
        return '(-0.001, 120.0]'
    elif resting_bp <= 128:
        return '(120.0, 128.0]'
    elif resting_bp <= 135.2:
        return '(128.0, 135.2]'
    elif resting_bp <= 145:
        return '(135.2, 145.0]'
    return '(145.0, 200.0]'


def get_cholesterol_cat(cholesterol):
    if cholesterol <= 84.999:
        return '[0, 0]'
    elif cholesterol <= 217:
        return '(84.999, 217.0]'
    elif cholesterol <= 263:
        return '(217.0, 263.0]'
    return '(263.0, 603.0]'


def get_max_hr_cat(max_hr):
    if max_hr <= 103:
        return '(59.999, 103.0]'
    elif max_hr <= 115:
        return '(103.0, 115.0]'
    elif max_hr <= 122:
        return '(115.0, 122.0]'
    elif max_hr <= 130:
        return '(122.0, 130.0]'
    elif max_hr <= 138:
        return '(130.0, 138.0]'
    elif max_hr <= 144:
        return '(138.0, 144.0]'
    elif max_hr <= 151:
        return '(144.0, 151.0]'
    elif max_hr <= 160:
        return '(151.0, 160.0]'
    elif max_hr <= 170:
        return '(160.0, 170.0]'
    return '(170.0, 202.0]'
