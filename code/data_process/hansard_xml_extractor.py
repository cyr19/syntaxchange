import collections

import tqdm
from bs4 import BeautifulSoup
import os
import zipfile
import re
import pandas as pd
from pathlib import Path
import glob


def decompress(zip_path):
    f = zipfile.ZipFile(zip_path, 'r')
    xml = f.read(f.namelist()[0])
    return xml


def read_xml(xml_file):
    soup = BeautifulSoup(xml_file, "xml")
    return soup


def date_year_matches(year, date_text, path):
    year_range = path.split('/')[2].split('-')
    lower, upper = int(year_range[0]), int(year_range[1][:4])
    # Tuesday, 16th February, 1909.
    #text_year = date_text[-5:-1]
    #try:
    text_year = re.search("1\d{3}", date_text)#.group(0)
    if text_year is None:
        return year, True
    #except:
    #print(date_text)
    #print(type(text_year[0]))
    #if text_year[0] != '1':
    #    final_year = year
    #assert text_year[0] == '1', f"{path} --- {date_text}"
    text_year_int = int(text_year.group(0))
    if lower <= text_year_int <= upper:
        final_year = str(text_year_int)
    #elif int(year) >= lower and int(year) <= upper:
    #    final_year = year
    else:
        final_year = year
    return final_year, year == str(text_year_int)


def simple_preprocess(text):
    new_text = re.sub(r"\n+\d+\n+", " ", text)  # "\n207\n"
    new_text = re.sub(r"\n{2,}", "\n", new_text)
    new_text = re.sub(r"\s{2,}", " ", new_text)
    new_text = re.sub(r"^\d+\.", "", new_text)  # "3.Mr..."
    # "THE EARL OP CRAWFORD: My Lords, I
    splits = new_text.split(': ')
    if len(splits) > 1 and re.match(r"^([A-Z]+\s*)+", splits[0]):
        new_text = ': '.join(splits[1:])
    return new_text


def read_house(soup, path, out_dir):
    houses = {}
    houses['lords'] = soup.find_all('houselords')
    houses['commons'] = soup.find_all('housecommons')
    zip_path = '/'.join(path.split('/')[-3:])

    for chamber, c_houses in houses.items():
        for i, house in enumerate(c_houses):
            dates = house.find_all('date')
            debates = house.find_all('debates')
            if len(debates) == 0:
                continue
            # multiple dates found and it's not the last section which follows by an answer section that contains another date
            if (len(dates) != 1) and (i != len(c_houses)-1):
                with open(os.path.join(out_dir, 'multiple_dates.txt'), 'a') as f:
                    f.write(f"{path}\n{'|'.join([d.get('format') for d in dates])}\n\n")

            date = dates[0].get('format').strip()
            date_text = dates[0].text.strip()
            #raise ValueError
            year, month, day = date.split('-')
            # check if the date matches the following text of the date
            corrected_year, matches = date_year_matches(year, date_text, path)

            if not matches:
                with open(os.path.join(out_dir, 'mismatched_dates.txt'), 'a') as f:
                    f.write(f"{path}\n{date+'|'+date_text}\n\n")
            date = '-'.join([corrected_year, month, day])
            decade = year[:3]+'0s'
            out_dir_tmp = os.path.join(out_dir, f"{decade}/{corrected_year}/")
            if not os.path.exists(out_dir_tmp):
                os.makedirs(out_dir_tmp)
            out_path = os.path.join(out_dir_tmp, f"{month}.csv")

            to_store = collections.defaultdict(list)
            for debate in debates:
                sections = debate.find_all('section')
                for section in sections:
                    titles = section.find_all('title')
                    title = '|'.join([t.text for t in titles])
                    texts = section.find_all('p')
                    #texts = [t.text for t in texts]
                    texts = [simple_preprocess(t.text) for t in texts]
                    size = len(texts)
                    to_store['chamber'] += [chamber]*size
                    to_store['date'] += [date]*size
                    to_store['section'] += [title]*size
                    to_store['text'] += texts
                    to_store['zip_path'] += [zip_path]*size

            df = pd.DataFrame(to_store)
            if len(df) == 0:
                print(path)
            if not os.path.exists(out_path) or os.stat(out_path).st_size<2:
                df.to_csv(out_path, header=True, index=False, sep='\t', mode='w')
            else:
                df.to_csv(out_path, header=False, index=False, sep='\t', mode='a')


if __name__ == '__main__':

    out_dir = 'data/new_collected_hansard_csv_v1/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #for dir in ['1803-1820', '1820-1830', '1830-1891', '1892-1908', '1909-1981(commons)', '1909-2004(lords)', '1981-2004(commons)']:
    #for dir in ['1909-1981(commons)', '1909-2004(lords)', '1981-2004(commons)']:
    for dir in ['1803-1820']:
        zip_files = glob.glob(f"data/new_collected_hansard/{dir}/*/*.zip")
        # remove redundant files
        read = []
        for zip_path in tqdm.tqdm(zip_files):
            file = zip_path.split('/')[-1]
            if file in read:
                continue
            read.append(file)
            xml_file = decompress(zip_path)
            soup = read_xml(xml_file)

            read_house(soup, zip_path, out_dir)



