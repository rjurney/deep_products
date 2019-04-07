#!/usr/bin/env python

import sys, os, re
import glob
import json
import random
from multiprocessing.dummy import Pool as ThreadPool
from datetime import date

import lxml
from lxml import etree

"""
EXTRACT DOCS FROM XML SCRIPT

I am a script that converts a few fields of USPTO patent application XML documents into JSON documents. I accept a single argument, the path
to a list of paths to XML files, each describing one patent application, to be processed.

I use Python multiprocessing/pids to utilize 120-220% CPU utilization per process. You can instantiate me to run in parallel with other pids
like me like so:

# Where /tmp/files holds .txt files with one path on each line
for f in /tmp/files/*.txt
do
    ./classifier/extract_docs_from_xml.py $f &
done
"""

class ParseException(Exception):
    """XML Parsing exception!"""
    pass


def jsonize_xml(dir_path):
    """Turn big XML documents into trimmed JSON documents."""

    try:
        root = etree.parse(dir_path).getroot()
        application_id = root.find('./us-bibliographic-data-application/application-reference/document-id/doc-number').text
        title = root.find('./us-bibliographic-data-application/invention-title').text.strip()
        abstract = ' '.join([i for i in root.find('./abstract').itertext()]).strip()
        description = ' '.join([i for i in root.find('./description').itertext()]).strip()
        raw_date = root.find('./us-bibliographic-data-application/application-reference/document-id/date').text
        date = '{}-{}-{}'.format(raw_date[0:4], raw_date[4:6], raw_date[6:8])

    except (AttributeError, lxml.etree.XMLSyntaxError) as e:
        raise ParseException()
    
    record = {
        'application_id': application_id,
        'date': date,
        'title': title,
        'abstract': abstract,
        'description': description
    }
    
    return record

def jsonize_xml_batch(date_paths):
    """Convert a list of file paths to aggregated JSON Lines files."""

    file_paths = date_paths['file_paths']
    ts = date_paths['timestamp']

    file_name_hash = '%08x' % random.getrandbits(64)

    with open('data/applications/json/{}.{}.jsonl'.format(ts, file_name_hash), 'w') as f_out:

        for file_path in file_paths:
            try:
                application = jsonize_xml(file_path)

                # For god's sakes stream your writes!
                f_out.write(
                    json.dumps(application) + '\n'
                )
                
            except ParseException as e:
                sys.stderr.write('.')
                sys.stderr.flush()

    print('Wrote out applications in {} file...'.format(file_name_hash))

    return True

def main():
    """Convert all patent application files from XML to JSON Lines."""

    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        sys.stderr.write('Error: must supply argument, a valid path to a file containing a list of paths to XML files!\n')
        exit()
    
    file_paths = [line.strip() for line in open(sys.argv[1])]
    if not file_paths and isinstance(file_paths, list) and len(file_paths) > 0:
        sys.stderr.write('Error: failed to read list of XML file paths!\n')
        exit()
    
    ts = date.today().isoformat()
    
    path_len = int((len(file_paths) / 10.0 + 1))
    path_chunks = [file_paths[i:i + path_len] for i in range(0, len(file_paths), path_len)]
    date_paths = [{'file_paths': path_chunk, 'timestamp': ts} for path_chunk in path_chunks]

    pool = ThreadPool(10)
    applications = pool.map(jsonize_xml_batch, date_paths)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
