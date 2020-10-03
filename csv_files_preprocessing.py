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
    df_merged = pd.read_csv(all_filenames[0], sep=',')
    for filename in all_filenames:
        if filename == all_filenames[0]:
            continue
        df = pd.read_csv(filename, sep=',')
        df_merged = pd.merge(df_merged, df, on='Date', how='left')
    df_merged.to_csv('merged_data.csv')
        
if __name__ == '__main__':
    main()