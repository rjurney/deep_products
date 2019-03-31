#!/usr/bin/env  bash

function split () {

    infile = $1

    BASE_NAME=`basename ${infile%.zip}`
    echo "#BASE_NAME: $BASE_NAME"

    echo "mkdir -p $BASE_NAME"
    mkdir -p $BASE_NAME

    cp $BASE_NAME.zip $BASE_NAME/
    cd $BASE_NAME

    # Unzip into the .xml file
    unzip $BASE_NAME.zip

    # Split the multi-xml file into multiple xml files
    csplit -f $BASE_NAME -b '.%05d.out.xml' -s -z $BASE_NAME.xml '/<?xml version="1.0" encoding="UTF-8"?>/' {*}

    # Cleanup the original XML file
    rm $BASE_NAME.xml

    # Lint all XML files
    for f in ipa*.out.xml;
    do
        mv "$f" "$f.bak" || exit 1
        xmllint --format "$f.bak" > "$f"
        rm "$f.bak"
    done

    gzip *.xml
    rm $BASE_NAME.zip

    cd ..
}

# Copy the file into the subdirectory and go there
for infile in ipa*.zip;
do
    split $infile &
done


# Cleanup
# rm $BASE_NAME.zip
