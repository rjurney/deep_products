#!/usr/bin/env python

import sys, os, re
from datetime import datetime, timedelta
import iso8601
import subprocess

if len(sys.argv) < 2:
    print('Must supply output directory name!')
    exit()

dir_path = sys.argv[1]
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

BASE_URL = 'https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/{}/ipa{}.zip'

this_week = iso8601.parse_date('2008-01-03')

while this_week < iso8601.parse_date('2018-12-01'):

    full_url = BASE_URL.format(
        this_week.year, 
        this_week.strftime('%y%m%d')
    )

    subprocess.run(['wget', '-P', dir_path, full_url])

    this_week = this_week + timedelta(weeks=1)
