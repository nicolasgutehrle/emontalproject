from gallica.gallica_interface import GallicaInterface
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Downloads XML file from given Gallica URL")
    
    parser.add_argument('Database',
                       metavar='db',
                       type=str,
                       help='Name of the folder where the data is saved')
    
    parser.add_argument('URL',
                       metavar='url',
                       type=str,
                       help='Url to the Gallica collection to scrap')  
    args = parser.parse_args()
    
    database = args.Database
    url = args.URL
    
    print(f'Saving in {database}')
    print(f'Collecting from {url}')
    
    
    interface = GallicaInterface(database, database_type='folder')

    # interface.retrieve_documents(url_fc)
    # interface.retrieve_ocr()
    # interface.retrieve_images()