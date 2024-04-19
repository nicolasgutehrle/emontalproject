from tqdm import tqdm
from glob import glob
import os
from multiprocessing.dummy import Pool
# import multiprocessing
import multiprocessing as mp
from main.ocrtextprocessor import OCRTextProcessor
import time
import argparse


def process_file_map(filepath):
    print(filepath)
    outpath = filepath.replace('.xml', '_clean.xml')
    clean_file = textcleaner.process(filepath)

    with open(outpath, 'w', encoding='utf-8') as f:
        print(f'Saving {outpath}')
        f.write(clean_file)
    
    return outpath

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Cleans OCR in given folder database")
    
    parser.add_argument('Database',
                       metavar='db',
                       type=str,
                       help='Name of the folder where the data is saved')
    
    parser.add_argument('LG',
                       metavar='lg',
                       type=str,
                       help='Language of the document. Currently supported: fr')  
    args = parser.parse_args()
    
    path = args.Database
    lg = args.LG
    
    print(f'Processing {path}')
    
    textcleaner = OCRTextProcessor(lg)

    file_list = []
    for folder in tqdm(glob(f'{path}/**', recursive=True)):
        if os.path.isdir(folder):
            foldername = os.path.basename(folder)
            if foldername.startswith('bpt'):

                if not os.path.exists(f'{folder}/{foldername}_clean.xml'):

                    file = f'{folder}/{foldername}.xml'
                    # print(file)
                    file_list.append(file)

    print('Number of files to clean: ', len(file_list))

    with Pool(mp.cpu_count() - 2) as p:
       results = p.map(process_file_map, file_list)
    
    # for file in file_list:
    #     process_file_map(file)
    #     break
