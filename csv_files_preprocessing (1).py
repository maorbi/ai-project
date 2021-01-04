import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def main():
    '''os.chdir("C:\\Users\\Shaked\\Downloads\\AI_PROJECT_BACKUP\\result1")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    for filename in all_filenames:
        df = pd.read_csv(filename, sep=',')
        filename_mod = filename[0 :len(filename)-4]
        df.columns = ['Date', 'Open_' + filename_mod, 'High_' + filename_mod,
                      'Low_' + filename_mod, 'Close_' + filename_mod,
                      'Adj Close_' + filename_mod, 'Volume_' + filename_mod]
        df.to_csv(filename, index=False)
    df_merged = pd.DataFrame()
    df_merged = pd.read_csv('^GSPC.csv', sep=',')
    for filename in all_filenames:
        if filename == '^GSPC.csv':
            continue
        df = pd.read_csv(filename, sep=',')
        df_merged = pd.merge(df_merged, df, on='Date', how='left')
    os.chdir("C:\\Users\\Shaked\\Downloads\\AI_PROJECT_BACKUP")
    df_merged = df_merged.dropna(axis='columns')
    df_polarity = pd.read_csv('Polarity.csv', sep=',')
    df_merged = pd.merge(df_merged, df_polarity, on='Date', how='left')
    #for i in range(len(df_merged['Close_^GSPC'])):
    new_column = df.iloc[:, -1:] + 1
    df_merged['s&p_original'] = new_column
    df_merged['s&p_original'] = df_merged['Close_^GSPC']
    df_merged['Close_^GSPC'] = smooth(df_merged['Close_^GSPC'], 5)
    df_merged.to_csv('merged_data.csv', index=False) '''
    df = pd.read_csv("merged_data.csv", sep=',')
    new_column = df.iloc[:, -1:] + 1
    df['trend'] = new_column
    df['trend'][0] = 0
    for i in range(len(df['trend']) - 1):
        if df['Close_^GSPC'][i] < df['Close_^GSPC'][i + 1]:
            df['trend'][i + 1] = 1
        else:
            df['trend'][i + 1] = 0
    new_column = df.iloc[:, -1:] + 1
    df['trend_neg'] = new_column
    for i in range(len(df['trend_neg'])):
        df['trend_neg'][i] = 1-df['trend'][i]
    df.to_csv('merged_data.csv', index=False)
    new_column = df.iloc[:, -1:] + 1
    df['avg_trend'] = new_column
    #file = open("data_merged.csv", "w+r")

    for i in range(len(df['avg_trend'])):
        if i < 8:
            df['avg_trend'] = 0

        else:
            if len(df['avg_trend']) - i < 7:
                pass
                if moving_average(df['s&p_original'][i-7]) < df['s&p_original'][i]:
                    df['avg_trend'][i] = 0
                else:
                    df['avg_trend'][i] = 1


def moving_average(iterator, window=7):
    avg = 0
    for i in range(window):
        avg += iterator[i]
    avg = avg/window
    return avg


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve((np.ndarray.flatten(y.to_numpy())), box, mode='same')
    return y_smooth


if __name__ == '__main__':
    main()
