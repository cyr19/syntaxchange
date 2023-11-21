import pandas as pd
import re
import glob
import os

import tqdm


def simple_preprocess(df, drop_all_duplicated_texts=False):
    #print(len(df)
    df = df[~df.text.isna()]
    if drop_all_duplicated_texts:
        df.drop_duplicates(inplace=True, subset='text')
    else:
        df.drop_duplicates(inplace=True)
    #print(len(df))
    texts = []
    for text in df['text']:
        #print(text)

        try:
            text = text.strip()
        #if '\n207\n' in text:
        #    print(text)
        except:
            print(text)
            print(df)
            raise ValueError

        new_text = re.sub(r"\n+\d+\n+", " ", text) #"\n207\n"
        new_text = re.sub(r"\n{2,}", "\n", new_text)
        new_text = re.sub(r"\s{2,}", " ", new_text)
        new_text = re.sub(r"^\d+\.", "", new_text) #"3.Mr..."
        #new_text = re.sub(r"^('[H.L.] ')+", "", new_text) #"3.Mr..."
        new_text = new_text.replace("[H.L.] ", '')

        # "THE EARL OP CRAWFORD: My Lords, I
        splits = new_text.split(': ')
        if len(splits) > 1 and re.match(r"^([A-Z]+\s*)+", splits[0]):
            new_text = splits[1]
            #print(new_text)
        #new_text = re.sub(r"^([A-Z]+\s*)+: ", "", new_text)
        #if '\n207\n' in text:
            #print(new_text)
        texts.append(new_text)
    df['text'] = texts
    return df


def drop_duplicate_and_move(df, old_path, drop_all_duplicated_texts=False):
    df = simple_preprocess(df, drop_all_duplicated_texts=drop_all_duplicated_texts)
    #decade = old_path.split('/')[3]
    decade, year = old_path.split('/')[3:5]
    new_dir = os.path.join(cleaned_data, decade, year)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    new_path = os.path.join(cleaned_data, old_path.split('csv/')[1])

    df.to_csv(new_path, index=False, header=True, mode='w', sep='\t')


# correct year
def correct_years(drop_all_duplicated_texts=False):
    logs = dirty_data + "check_log.txt"
    # print()
    # print(glob.glob(dirty_data+'*/'+logs))
    logs = glob.glob(logs) + glob.glob(dirty_data + '*/check_log.txt')
    #print(logs)
    for log in logs:
        #print(log)
        dir = '/'.join(log.split('/')[:-1])
        #print(dir)
        csv_files = glob.glob(dir + '/*/*.csv') + glob.glob(dir + '/*.csv')
        with open(log, 'r') as f:
            log_text = f.readline().strip()

        #print(log_text)
        corrected_years = log_text.split(',')
        assert len(corrected_years) == len(csv_files), csv_files
        if ':' in corrected_years[0]:
            years = {}
            for y in corrected_years:
                month, corrected_y = y.split(':')
                years[month] = corrected_y
           # print(csv_files[0])
            months = [f[-6:-4] for f in csv_files]
            #print(months)
            corrected_years = [years[m] for m in months]
        #print(corrected_years)
        #print(csv_files)
        #print()

        for year, csv in zip(corrected_years, csv_files):
            df = pd.read_csv(csv, delimiter='\t')
            month = csv[-6:-4]

            month_day = ['-'.join(d.split('-')[-2:]) for d in df['date']]
            df['date'] = [year + '-' + month_day[i] for i in range(len(df))]
            # print(month)
            # out_path = os.path.join(cleaned_data, year[:3]+'0s', year, csv.split('/')[-1])
            out_path = os.path.join(cleaned_data, year[:3] + '0s', year)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            out_path = os.path.join(out_path, csv.split('/')[-1])
            df = simple_preprocess(df, drop_all_duplicated_texts=drop_all_duplicated_texts)

            if os.path.exists(out_path):
                df.to_csv(out_path, index=False, sep='\t', mode='a', header=False)
            else:
                df.to_csv(out_path, index=False, sep='\t', mode='w', header=True)


if __name__ == '__main__':

    # drop duplicate

    dirty_data = "data/new_collected_hansard/csv/*/"
    #HOME = "data/new_collected_hansard/"
    cleaned_data = "data/new_collected_hansard/csv_cleaned/1950-2004/"
    #if not os.path.exists(cleaned_data):
    #    os.makedirs(cleaned_data)
    csv_files = glob.glob(dirty_data+"*/*.csv")
    drop_all_duplicated_texts = False

    for csv_file in tqdm.tqdm(csv_files):
        year = csv_file.split('/')[4]
        #if int(year) < 1909 or int(year) > 2004 or year[0]==' ':
        #if int(year) < 1950 or int(year) > 2004 or year[0]==' ' or '198' in year:
        if int(year) < 2000 or int(year) > 2004 or year[0] == ' ':
            continue
        try:
            df = pd.read_csv(csv_file, delimiter='\t')
            df = df[df.zip_path != "zip_path"]
        except:
            with open(csv_file , 'r') as f:
                s = f.read()
            header = 'chamber	date	section	text	zip_path'
            new = header
            new += '\n'.join(s.split('chamber	date	section	text	zip_path')[1:])
            with open(csv_file , 'w') as f:
                f.write(new)
            df = pd.read_csv(csv_file, delimiter='\t')


        print(df)
        # fix col mismatch

        i = 5
        cols = df.columns
        if 'Unnamed: 0' in cols:
            print(df)
            df_wo_index = df[df.zip_path.isna()].copy(deep=True)
            while i > 0:
                df_wo_index[cols[i]] = df_wo_index[cols[i-1]]
                #print(i)
                i-=1
            df_wo_index['Unnamed: 0'] = [0]*len(df_wo_index)
            #df_wo_index['zip_path']
            df_w_index = df[~df.zip_path.isna()]
            df = pd.concat([df_w_index, df_wo_index], ignore_index=True)
            df.drop(columns='Unnamed: 0', inplace=True)
        #print(df)
        #print(df['zip_path'].isnull().values.any())

        #df.drop(columns='Unnamed: 0', inplace=True)
        #print(df)
        df = df[df.zip_path != "zip_path"]
        print(df)

        drop_duplicate_and_move(df, csv_file, drop_all_duplicated_texts)
    correct_years(drop_all_duplicated_texts)
        #break
