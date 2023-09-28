# This script packages up the files and directories we want to include in 
# a zip file for submission
files="README.md requirements.txt run_chosen_hyps.sh run.sh setup.py smoke_test.sh"
directories="notebooks prompt_risk scripts tgi"
zip_file="submission.zip"

# Remove the zip file if it already exists
if [ -f $zip_file ]; then
    rm $zip_file
fi

# Create the zip file
zip -r $zip_file $files $directories

# Check that the zip file was created
if [ ! -f $zip_file ]; then
    echo "Error: $zip_file was not created"
    exit 1
fi

# Check that the zip file is not empty
if [ ! -s $zip_file ]; then
    echo "Error: $zip_file is empty"
    exit 1
fi

# Check that the zip file contains all the files and directories we want
for file in $files; do
    if ! zipinfo $zip_file | grep -q $file; then
        echo "Error: $zip_file does not contain $file"
        exit 1
    fi
done

for directory in $directories; do
    if ! zipinfo $zip_file | grep -q $directory; then
        echo "Error: $zip_file does not contain $directory"
        exit 1
    fi
done

echo "Success: $zip_file was created"
exit 0