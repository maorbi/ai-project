import pandas as pd
import os
import glob

def main():
    os.chdir("C:\\Users\\user\\Desktop\\project data\\‏‏result1")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    for filename in all_filenames:
        df = pd.read_csv(filename, sep=',')
        filename_mod = filename[0 :len(filename)-4]
        df.columns = ['Date', 'Open_' + filename_mod,  'High_' + filename_mod,
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
    os.chdir("C:\\Users\\user\\Desktop\\project data")
    df_merged = df_merged.dropna(axis='columns')
    df_merged.to_csv('merged_data.csv', index=False)
        
if __name__ == '__main__':
    main()