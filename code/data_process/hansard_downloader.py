import os.path

from bs4 import BeautifulSoup
from pathlib import Path
import requests
import re
import wget


def download_zip(paths, out, page_num):
    for i, path in enumerate(paths):
        start = (page_num-1)*10+1
        end = start+9
        new_out = os.path.join(out, f"{start}-{end}")
        if not os.path.exists(new_out):
            os.makedirs(new_out)
        wget.download('http:'+path, out=new_out)


def collect_new_pages(url, page=None, soup=None, init=True):
    if init:
        page_response = requests.get(url)
        soup = BeautifulSoup(page_response.content, "lxml")
    else:
        postdata = {"__EVENTTARGET": page.split("\'")[1]}
        input_data = soup.find_all("input", {"type": "hidden"})
        for data in input_data:
            postdata[data["id"]] = data["value"]
        page_response = requests.post(url, data=postdata)
        soup = BeautifulSoup(page_response.content, "lxml")

    table = soup.find("table", {"class": "normaltext"})
    paths = table.find_all('a')
    zip_paths = [path.get('href') for path in paths if '.zip' in path.get('href')]
    new_page_patter = "^javascript:__doPostBack\('(.*)',''\)$"
    new_pages = [path.get('href') for path in paths if re.match(new_page_patter, path.get('href'))]
    if not init:
        new_pages = new_pages[1:]
    print('\n'.join(zip_paths))

    return soup, zip_paths, new_pages


def recursive_download(url, pages, soup, page_num, final=1):
    print('\nmajor page: ', page_num)
    for i, page in enumerate(pages):
        new_soup, zip_paths, new_pages = collect_new_pages(url, page, soup, init=False)
        download_zip(paths=zip_paths, out=out_dir, page_num=page_num)
        if i == len(pages) - 1:
            if page_num == final:
                return '\nfinished!!!!'
            else:
                page_num += 1
                return recursive_download(url, new_pages, new_soup, page_num, final=final)


def download(url, out_dir, final=1):
    soup, zip_paths, new_pages = collect_new_pages(url, out_dir, init=True)
    download_zip(paths=zip_paths, out=out_dir, page_num=1)
    done = recursive_download(url, new_pages, soup, page_num=1, final=final)
    print(done)


if __name__ == '__main__':
    #url = "https://www.hansard-archive.parliament.uk/Official_Report,_House_of_Commons_(5th_Series)_Vol_1_(Jan_1909)_to_Vol_1000_(March_1981)"
    #url = "https://www.hansard-archive.parliament.uk/The_Official_Report,_House_of_Lords_(5th_Series)_Vol_1_(Jan_1909)_to_2004"
    #out_dir = 'data/new_collected_hansard/1909-2004(lords)'
    #print(f'collecting {out_dir}')
    #download(url, out_dir)

    #url = "https://www.hansard-archive.parliament.uk/The_Official_Report,_House_of_Commons_(6th_Series)_Vol_1_(March_1981)_to_2004"
    #out_dir = 'data/new_collected_hansard/1981-2004(commons)'
    #print(f'collecting {out_dir}')
    #download(url, out_dir)

    url = "https://www.hansard-archive.parliament.uk/Parliamentary_Debates_(3rd_Series)_Vol_1_(Oct_1830)_to_Vol_356_(August_1891)"
    out_dir = 'data/new_collected_hansard/1830-1891'
    print(f'collecting {out_dir}')
    download(url, out_dir, final=5)


