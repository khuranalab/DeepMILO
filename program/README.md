## Requirement:
+ Dependency packages: tensorflow 1.7, keras 2.2, h5py, numpy
+ Homo_sapiens.GRCh37.75.dna.primary_assembly.fa (downloaded from ensembl) in the current folder
+ you can install the libraries from requirements.txt using the following command:

  `$ sudo pip install -r requirements.txt`


## Folder content:
* model: contains the loop model
* input: contains sample input files (simple somatic mutation from ICGC)
* loopDB:
	* all_insulator_loop.bed: list of constitutive loops
	* all_loop_probabilities.txt: probability of insulator loops without mutations 	
	
			


## Running
### Input: The program requires 2 input files:
  + a simple somatic mutation file from ICGC (in .tsv format)
  + a structural variant file from ICGC (in .tsv format)


### Output: 2 folders
  + one folder contains sequence data for patients taking into consideration mutations from patients 
  + another folder contains loop probability predictions for each patient

### Execute:  There are 3 options

  + `-ssm` to specify simple somatic mutation file
  + `-sv` to specify structural variant file
  + `-loop` to specify loops used to predict impacts of variants, when `-loop` is specified but `-ssm` and `-sv` are not specified, the program will calculate loop probability for loops 

#### Examples:
  
	
`$ python main.py -ssm input/simple_somatic_mutation.open.tsv` # if only ssm is available
  	
`$ python main.py -ssm input/ssm.ts -sv input/sv.tsv` # if both ssm and sv are available
	
`$ python main.py -loop loopDB/all_insulator_loop.bed`  # to caculate loop probability without mutation for loops 

	



