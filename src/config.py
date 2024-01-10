image_dir = "/data/datasets/ODIR/RPS_analysis/normal_fundus/"
results_dir = "/data/datasets/ODIR/RPS_analysis/rps_results/"


worker = 0  # number of workers
batch_size = 4
device = "cuda"  # can specify cuda:[PCID] or 'cuda' or 'CPU'
sample_num = False  # Put False if you do not want to sample, otherwise specify the number (int) of random samples you want to take from your dataset
sparse = True  # if False then saves all the pre-processed images from step M1
debug = False  # save debug files from extract pigmentation
quality_thresh = (
    "good"  # options are "good", "usable", "all" to use for thresholding the quality
)
