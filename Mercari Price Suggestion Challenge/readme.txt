Follow the following steps:

1. Extract the Zip folder to a location on your C drive.
2. The root folder contains following documents:
	a. Problem Statement.pdf
	b. Project Report.pdf
3. The root folder contains following folders:
	a. script
		i. This folder contains following scipts:
			1. analysis.R -- for analysing the datasets
			2. keras_tensorflow.py --script containing model for prediction
			3. tensorflow self test.py --self test script for diagnosing tensorflow installation
	b. input
		i. This folder contains the provided datasets
	c. output
		i. This folder contains following .csv files
			1. sample_submission.csv --prediction for test.tsv using keras model
			2. sample_submission_stg2.csv --prediction for test_stg2.tsv using keras model
4. To run the model:
	a. Open Anaconda Prompt
	b. Navigate to the script folder
	c. run the following commands:
		i. set "KERAS_BACKEND=tensorflow"
		ii. python.exe keras_tensorflow.py
5. To diagnose tensorflow installation run the following script:
	a. python.exe tensorflow self test.py
6. Download test_stg2.tsv from kaggle website and put in input folder