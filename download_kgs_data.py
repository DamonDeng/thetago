

from data_loader.index_processor import KGSIndex

print('Started to download KGS data...')

index = KGSIndex(data_directory='data')
index.download_files()