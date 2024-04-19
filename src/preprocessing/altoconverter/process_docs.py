from main.processors import Processor
import argparse
from tqdm import tqdm
from glob import glob
import pandas as pd
from multiprocessing.dummy import Pool
import multiprocessing as mp
import os

def process_file_map(filepath):
    print(filepath)
    
    processed_doc = proc.convert_doc(filepath)
    
    docbook = processed_doc['docbook'].prettify()
    line_metadata = processed_doc['line_metadata']
    block_metadata = processed_doc['block_metadata']
    
    basename = os.path.dirname(filepath)
    filename = os.path.basename(filepath)[:-4]
    
    xmloutpath = f'{basename}/{filename}'
    xmloutpath = xmloutpath.replace('_clean', '_docbook.xml')
    
    with open(xmloutpath, 'w', encoding='utf-8') as f:
        print(f'Saving {xmloutpath}')
        f.write(docbook)    
    
    lineoutpath = f'{basename}/line_metadata.csv'
    line_metadata.to_csv(lineoutpath, index=False)
    print(f'Saving {lineoutpath}')
    
    blockoutpath = f'{basename}/block_metadata.csv'
    block_metadata.to_csv(blockoutpath, index=False)
    print(f'Saving {blockoutpath}')

    print()

    return True

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Process documents in database folder for Logical Layout Analysis, Article segmentation and Text Readability, and saves them in a Docbook XML format")
    
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
    
    proc = Processor(lg)
    file_list = []
    for folder in tqdm(glob(f'{path}/**', recursive=True)):
        if os.path.isdir(folder):
            foldername = os.path.basename(folder)
            if foldername.startswith('bpt'):

                if not os.path.exists(f'{folder}/{foldername}_docbook.xml'):

                    file = f'{folder}/{foldername}_clean.xml'
                    file_list.append(file)
                    
    print('Number of files to clean: ', len(file_list))

    for file in file_list:
        process_file_map(file)

#    with Pool(mp.cpu_count() - 2) as p:
 #      results = p.map(process_file_map, file_list)


