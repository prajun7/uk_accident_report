#!/bin/bash

# get 2005 - 2014 data (one file) and 2015 data.
# unpack all, remove 1st line of 2015 data and append it to 2005-14 data

#download to zips dir
mkdir zips
#wget http://data.dft.gov.uk/road-accidents-safety-data/RoadSafetyData_2015.zip -P zips
#wget http://data.dft.gov.uk.s3.amazonaws.com/road-accidents-safety-data/Stats19_Data_2005-2014.zip -P zips

# unzip to cwd
echo Unzipping
unzip zips/RoadSafetyData_2015.zip
unzip zips/Stats19_Data_2005-2014.zip

# loose first line of 2015 data
echo Cleaning 1st line
sed -i '1d' Accidents_2015.csv
sed -i '1d' Casualties_2015.csv
sed -i '1d' Vehicles_2015.csv

# mung together
echo Concatenating
cat Accidents0514.csv Accidents_2015.csv >> Accidents0515.csv
cat Casualties0514.csv Casualties_2015.csv >> Casualties0515.csv
cat Vehicles0514.csv Vehicles_2015.csv >> Vehicles0515.csv

echo Cleanup
rm Accidents0514.csv Accidents_2015.csv
rm Casualties0514.csv Casualties_2015.csv
rm Vehicles0514.csv Vehicles_2015.csv

echo Done!