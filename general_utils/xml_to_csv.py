"""
Date Altered: 5/1/2019

@author: Eric Bianchi
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

#######################################################################################################################
def main(srcDirectory, destDirectory):
    # convert training xml data to a single .csv file
    print("converting xml training data . . .")
    trainCsvResults = xml_to_csv(srcDirectory)
    trainCsvResults.to_csv(destDirectory, index=None)
    print("training xml to .csv conversion successful, saved result to " + destDirectory)

# end main

#######################################################################################################################
def xml_to_csv(path):
    """

    Parameters
    ----------
    [path] : string
        path to xml directory.

    Returns
    -------
    [xml_df] : csv file
        csv file following bounding box information.

    """
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text,
                     int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
            xml_list.append(value)
        # end for
    # end for

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
# end function

#######################################################################################################################

