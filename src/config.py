image_dir = "/home/images/" #edit to your own directory path
results_dir = "/home/results/" #edit to your own directory batch
csv_path = False #edit to pass a csv with a column called 'path' which has the path to all images
worker = 0  # number of workers
device = "cuda"  # can specify cuda:[PCID] or 'cuda' or 'CPU'
sample_num = False  # Put False if you do not want to sample, otherwise specify the number (int) of random samples you want to take from your dataset
sparse = True  # if False then saves all the pre-processed images from step M1
debug = False  # save debug files from extract pigmentation
quality_thresh = (
    "good"  # options are "good", "usable", "all" to use for thresholding the quality
)

batch_size = 4 #global batchsize if you do not want to set individual bs, will not be replaced by step specific bs
batch_M1_quality = None # set this to optimize speed. it is a bs of 25 for 8GB of ram
batch_M2_vessel = None # set this to optimize speed it is around 7 for 8GB
batch_M2_disc = None #set this to optimize speed, it is around 35 for 8GB
