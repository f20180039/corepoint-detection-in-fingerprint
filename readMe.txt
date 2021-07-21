(Necessary folders to be present)
	   |data_used
	   |g_Truth
FUT phase1 |Mask_gt
           |Predictions				   |data_used 
	   |test-----------------------------------|g_Truth
	   |Training-------------|Loss_files	   |Mask_gt
	   |Visual_predictions	 |Plot
				 |Val_sample
				 |Weights

0)run cleaning_dataset.py# to remove files for which info is not complete
	input example:-
	...../FUT phase1

1)run Grountruth_in_one_file.py 
	input example:-
	...../FUT phase1
	
	outputs- gt_all_data.txt in the same folder where the python file is located(do not remove until the mask is generated)

2)run Code_for_mask.py
	input example:-
	...../FUT phase1
	
	output- generates mask and stores in the folder Mask_gt of the given directory

3)run split.py
	input example:-
	...../FUT phase1
	*Note1 - It modifies the original dataset(Cuts & Pastes the 20% in the test folder)
	
	*Caution - Run only once for a given dataset otherwise it reduces the training set by 20% every time its run.

4)run Hourglass?.py # ?={1,2,3,4}
	for training:
	#parser.add_argument('--phase', dest='phase', default='train', help='train, test') 
	input:
	train_folder = ...../FUT phase1
    	model_folder = ...../FUT phase1/Training

Note - Sometimes the y coordinates do not train(can be known by looking at the coordinate printed after every epoch) 
or if there is an error delete the Weights which might have been created and start training again. 
It usually works the second time around

	for testing:
	#parser.add_argument('--phase', dest='phase', default='test', help='train, test')
	input:
	test_input = ...../FUT phase1/test
    	model_folder = ...../FUT phase1/Training
    	test_folder = ...../FUT phase1
*Note - Before testing make sure the Predictions folder is empty and rename the folder specifying the train and test dataset used for future reference 

5)run compare.py
	input:
	path1=...../FUT phase1/test/g_Truth
	path2=...../FUT phase1/Predictions

	