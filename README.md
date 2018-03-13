# svd
Audio segmentation using energy


INTRODUCTION


SVD i.e. Singular Valuse Decomposition is applied on pooled time-frequency representation of bird vocalizations to learn basis vectors.By ustilizing only few of the bases,a compact feature representation is obtained for the test data.
After some simple post-processing, a threshold is used to reliably distinguish bird vocalizations from other sounds.


ABOUT CODE


Code works perfectly with  Python 2 (2.7+, possibly also 2.6).
You can run the included svd.py script in the following format:

python svd.py testFilePath segmentedAudiosDirectoryPath

Number of training files are mentioned using numTrainFiles variable which can be modified accordingly i.e. 1 for using only one training file or 2 for using two training files or similarly 3.
Ensure training files are as clean as possible i.e. no background or overlapping noise is present. Since,these will be the files through which code will learn and segment audios from test file.

Path of training data is mentioned using trainPath1,trainPath2,trainPath3 variables,which can be commented out depending on number of training files.

The code internally performs checks to see if path to code,test file and segmentation directory is mentioned properly.

Please follow latest branch to get updated version of code.

