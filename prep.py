import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from math import sqrt
from sklearn.metrics import mean_squared_error
from neuralprophet import NeuralProphet, set_log_level

def time_series_split(df):
    train_size = .90
    n = df.shape[0]
    test_start_index = round(train_size * n)

    train = df[:test_start_index] # everything up (not including) to the test_start_index
    test = df[test_start_index:]

    train_size = .90
    n2 = train.shape[0]
    validate_start_index = round(train_size * n2)

    train = df[:validate_start_index] # everything up (not including) to the test_start_index
    validate = df[validate_start_index:test_start_index ]

    return train, validate, test


def train_test_only(df):
    # Reassign the sale_date column to be a datetime type
    df.Year = pd.to_datetime(df.Year)

    # Sort rows by the date and then set the index as that date
    df = df.set_index("Year").sort_index()

    train_size = .80
    n = df.shape[0]
    test_start_index = round(train_size * n)

    train = df[:test_start_index] # everything up (not including) to the test_start_index
    test = df[test_start_index:] # everything from the test_start_index to the end

    plt.plot(train.index, train.total_sale)
    plt.plot(test.index, test.total_sale)

    return train, test

def seasonality_plots(train, time):
    result = seasonal_decompose(train, model='ad', extrapolate_trend='freq', period=time)

    from pylab import rcParams
    rcParams['figure.figsize'] = 12,5
    result.plot()

def us_car_model(train, test):
    train['multiplicative'] = ExponentialSmoothing(train['total_sale'],trend='mul').fit().fittedvalues
    #train[['total_sale','multiplicative']].plot(title='Holt Winters Double Exponential Smoothing')
    fitted_model = ExponentialSmoothing(train['total_sale'],trend='mul',seasonal='mul',seasonal_periods=8).fit()
    validate_predictions = fitted_model.forecast(7)
    validate_predictions.index = test.index
    train['total_sale'].plot(legend=True,label='TRAIN')
    #validate['total_sales'].plot(legend=True,label='validate',figsize=(6,4))
    test['total_sale'].plot(legend=True,label='test',figsize=(6,4))
    validate_predictions.plot(legend=True,label='PREDICTION')
    plt.title('Train, validate and Predicted validate using Holt Winters')
    rmse = round(sqrt(mean_squared_error(test.total_sale, validate_predictions)), 0)
    print(f'The RMSE for the model is: {rmse}')

def car_forecast(train, test, years):
    fitted_model = ExponentialSmoothing(train['total_sale'],trend='mul',seasonal='mul',seasonal_periods=8).fit()
    forecast_predictions = fitted_model.forecast(years)
    
    forecast_predictions=pd.DataFrame(forecast_predictions)
    year =  [2014, 2015, 2016,2017,2018,2019,2020,2021, 2022, 2023,2024,2025, 2026, 2027, 2028,2029,2030]

    forecast_predictions['year'] = year

    forecast_predictions['year'] = pd.to_datetime(forecast_predictions['year'], format="%Y")

    forecast_predictions = forecast_predictions.set_index('year')

    train['total_sale'].plot(legend=True,label='TRAIN')
    
    test['total_sale'].plot(legend=True,label='test',figsize=(6,4))
    forecast_predictions[0].plot(legend=True,label='PREDICTION')
    plt.title('Train, validate and Predicted validate using Holt Winters')

def last_obvs(train, test):
    ## Assigning last observade value to a variable.
    last_sale = train['total_sale'][-1:][0]
    print(f'The last sale observation is: {last_sale}')
    yhat_df = pd.DataFrame(
        {'total_sale': [last_sale]},
        index=test.index)
    rmse = round(sqrt(mean_squared_error(test.total_sale, yhat_df.total_sale)), 0)
    print(f'The RMSE is: {rmse}')
    plt.figure(figsize = (12,4))
    plt.plot(train.total_sale, label = 'Train', linewidth = 1)
    plt.plot(test.total_sale, label = 'Validate', linewidth = 1)
    plt.plot(yhat_df.total_sale)
    plt.title('total sale') 




#### Jeremy Global Production PREP

def clean_copper(df):
    
    df = df.replace(',','', regex=True)

    df = df.replace('/p','', regex=True)

    df.year = df.year.astype(float)
    df.year = df.year.astype(int)

    df.set_index('year', inplace=True)
   
    df = df.astype('int')

    df = df.reset_index()

   

    return df


def copper_split(df):
    train_size = .90
    n = df.shape[0]
    test_start_index = round(train_size * n)

    train = df[:test_start_index] # everything up (not including) to the test_start_index
    test = df[test_start_index:]

    train_size = .90
    n2 = train.shape[0]
    validate_start_index = round(train_size * n2)

    train = df[:validate_start_index] # everything up (not including) to the test_start_index
    validate = df[validate_start_index:test_start_index ]


    return train, validate, test


def last_global(train, validate):
    # assign period of 1, representing one year
    period = 1

    # assign values for moving avg
    mine_prod = round(train['mine_production'].rolling(period).mean().iloc[-1], 2)
    refined_prod = round(train['refined_production'].rolling(period).mean().iloc[-1], 2)

    # create dataframe with these values
    yhat_df = pd.DataFrame({'mine_production': [mine_prod],
                            'refined_production': [refined_prod],
                           }, index = validate.index)

    rmse1 = round(sqrt(mean_squared_error(validate.mine_production, yhat_df.mine_production)), 0)
    print(f'The mine production RMSE is: {rmse1}')
    plt.figure(figsize = (12,4))
    plt.plot(train.mine_production, label = 'Train', linewidth = 1)
    plt.plot(validate.mine_production, label = 'Validate', linewidth = 1)
    plt.plot(yhat_df.mine_production)
    plt.title('total sale') 
     
    rmse2 = round(sqrt(mean_squared_error(validate.refined_production, yhat_df.refined_production)), 0)
    print(f'The refined production RMSE is: {rmse2}')
    plt.figure(figsize = (12,4))
    plt.plot(train.refined_production, label = 'Train', linewidth = 1)
    plt.plot(validate.refined_production, label = 'Validate', linewidth = 1)
    plt.plot(yhat_df.refined_production)
    plt.title('total sale') 


def global_prophet(train, validate):
    mine_prod_df = pd.DataFrame(train.mine_production)

    mine_prod_df = mine_prod_df.reset_index()

    mine_prod_df.rename(columns = {'year':'ds'}, inplace=True)

    mine_prod_df.rename(columns = {'mine_production':'y'}, inplace=True)

    mine_prod_df['ds'] = pd.to_datetime(mine_prod_df['ds'])

    # m = NeuralProphet()
    # metrics = m.fit(mine_prod_df)
    # forecast = m.predict(mine_prod_df)

    # fig_forecast = m.plot(forecast)

    # # clean validate for prophet
    # val_mine_prod_df = pd.DataFrame(validate.mine_production)
    # val_mine_prod_df = val_mine_prod_df.reset_index()
    # val_mine_prod_df.rename(columns = {'year':'ds', 'mine_production':'y'}, inplace=True)

    # # validate forecast
    # val_forecast = m.predict(val_mine_prod_df)

    # # validate plot
    # fig_forecast = m.plot(val_forecast)


    m4 = NeuralProphet(seasonality_mode= "auto", learning_rate = 0.1)
    metrics_train2 = m4.fit(mine_prod_df)
    future = m4.make_future_dataframe(mine_prod_df, periods=65, n_historic_predictions='auto')
    forecast = m4.predict(future)
    fig = m4.plot(forecast)
    mine_prod_df.y.plot()