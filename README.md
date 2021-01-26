# oco-project
Hogwild ! and other Classic Online Convex Optimization algorithms implementation for OCO course.

Raw data (`mnist_train.csv` and `mnist_test.csv`) can be downloaded from https://pjreddie.com/projects/mnist-in-csv/, and should be placed in the `data/raw/` folder.
Run `python src/data_utils.py` to load, normalize and save the data as numpy arrays to the disk, so they can be loaded faster afterwards.
You should now be able to train the algorithms implemented in `algos.py` by running `python src/algos.py`.
