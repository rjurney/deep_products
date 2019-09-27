#
# Given a stratification filter limit (ex. 50000 or 5000), fetch all the S3 data for that limit
#

# First required argument is the filter limit
if [ "$1" != "" ]; then
    LIMIT="$1"
else
    echo "Must specify stratification limit!"
    exit 1
fi

# Second optional argument is the data's date
if [ "$2" != "" ]; then
    DATA_DATE="$2"
else
    echo "No date specified. Using date '08-05-2019' as default date..."
    DATA_DATE="08-05-2019" 
fi

# Verify they have the s3cmd command
if [ `command -v s3cmd` == "" ]; then
    echo "s3cmd not found in PATH! Please install s3cmd! See: https://s3tools.org/download"
    exit 1
fi

BUCKET="stackoverflow-events"
BASE_URL="s3://${BUCKET}/${DATA_DATE}"
OUTPUT_DIR="data/stackoverflow/${DATA_DATE}/"


# Grab the files...
FETCH_URI="${BASE_URL}/Questions.Stratified.Final.${LIMIT}.parquet"

# Print all commands as they run
set -x

# Make the data directory...
mkdir -p "${DATA_OUTPUT_DIR}"

# Fetch the data...
s3cmd get --recursive ${FETCH_URI} ${OUTPUT_DIR}/
s3cmd get "${BASE_URL}/final_report.${LIMIT}.json" ${OUTPUT_DIR}/
s3cmd get "${BASE_URL}/index_tag.${LIMIT}.json" ${OUTPUT_DIR}/
s3cmd get "${BASE_URL}/tag_index.${LIMIT}.json" ${OUTPUT_DIR}/
s3cmd get "${BASE_URL}/label_counts.${LIMIT}.json" ${OUTPUT_DIR}/
s3cmd get "${BASE_URL}/sorted_all_tags.${LIMIT}.json" ${OUTPUT_DIR}/
