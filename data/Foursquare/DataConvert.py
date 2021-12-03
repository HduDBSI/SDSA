import argparse
import pandas as pd
import datetime
import time


def time_convert(time_str):
    try:
        time_list = time_str.split()
        time_list[4] = 'CST'
        time_str = ''
        for list_str in time_list:
            time_str = time_str + list_str + ' '
        time_str = time_str.strip()
        dt = datetime.datetime.strptime(time_str, "%a %b %d %X %Z %Y")
    except:
        return ''
    else:
        return dt


def location_convert(location_str):
    try:
        len1 = len(location_str)
        location_list = location_str[1:len1 - 1].split(',')
    except:
        print('error' + location_str)
        return [0, 0]
    else:
        if len(location_list) > 1:
            return location_list[0], location_list[1]
        else:
            return [0, 0]


def category_convert(category_str):
    try:
        return category_str[1:len(category_str) - 1]
        # return category_str.split(',')
    except:
        return ''


class Data:
    def __init__(self, file_path, file_name):
        if file_name == 'checkin_venues.txt':
            self.data = pd.read_csv(
                file_path + file_name,
                header=1,
                sep="\t",
                names=['userID', 'Time', 'VenueId', 'VenueLocation', 'VenueCategory']
            )
        elif file_name == 'dataset_TSMC2014_NYC.txt' or file_name == 'dataset_TSMC2014_TKY.txt':
            self.data = pd.read_csv(
                file_path + file_name,
                header=None,
                index_col=None,
                sep="\t",
                encoding='iso-8859-1',
                names=['userID', 'VenueId', 'VenueCategoryId', 'VenueCategory', 'lat', 'lon', 'TimeZone', 'Time']
            )
            print(len(self.data['userID'].unique().tolist()))

    def convert(self):
        self.data['new_time'] = self.data.apply(lambda row: time_convert(row['Time']), axis=1)
        self.data['lat'] = self.data.apply(lambda row: location_convert(row['VenueLocation'])[0], axis=1)
        self.data['lon'] = self.data.apply(lambda row: location_convert(row['VenueLocation'])[1], axis=1)
        self.data['new_category'] = self.data.apply(lambda row: category_convert(row['VenueCategory']), axis=1)

    def convert_TSMC2014(self):
        self.data['new_time'] = self.data.apply(lambda row: time_convert(row['Time']), axis=1)

    def save_to_txt(self, save_path, output_name):
        # with open(save_path + "Foursquare_test.txt", "w+") as f:
        #     for index, row in self.data.iterrows():
        #         f.write(row['userID'].astype(str) + '\t'
        #                 + row['new_time'].astype(str) + '\t'
        #                 + row['lat'].astype(str) + '\t'
        #                 + row['lon'].astype(str) + '\t'
        #                 + row['VenueId'].astype(str) + '\t'
        #                 + row['new_category'].astype(str) + '\n')
        if output_name == 'Foursquare_output.txt':
            self.data.to_csv(save_path + output_name, index=0, columns=['userID', 'new_time', 'VenueId', 'lat', 'lon',
                                                                        'new_category'])
        elif output_name == 'TSMC2014_NYC_output.txt' or output_name == 'TSMC2014_TKY_output.txt':
            self.data.to_csv(save_path + output_name, index=0, columns=['userID', 'new_time', 'VenueId', 'lat', 'lon',
                                                                        'VenueCategory'])


if __name__ == '__main__':
    # file_path = '/home/shenyi/Data/raw_data/'
    # file_name = 'checkin_venues.txt'
    # save_path = '/home/shenyi/Data/raw_data/output/'
    # output_name = 'Foursquare_output.txt'
    file_path = '/home/shenyi/Data/dataset_tsmc2014/'
    # file_path = '/Users/shenyi/Desktop/Data/dataset_tsmc2014/'
    file_name = 'dataset_TSMC2014_NYC.txt'
    save_path = '/home/shenyi/Data/dataset_tsmc2014/output/'
    # save_path = '/Users/shenyi/Desktop/Data/dataset_tsmc2014/output/'
    output_name = 'TSMC2014_NYC_output.txt'
    data = Data(file_path, file_name)
    # data.convert()
    data.convert_TSMC2014()
    data.save_to_txt(save_path, output_name)
