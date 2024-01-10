<<<<<<< HEAD
image_dir = "/home/images/" #edit to your own directory path
results_dir = "/home/results/" #edit to your own directory batch
>>>>>>> 48bd5af6076aab1ce184d8a086590b9a9b5a0fe0

worker = 0  # number of workers
batch_size = 4
device = "cuda"  # can specify cuda:[PCID] or 'cuda' or 'CPU'
sample_num = False  # Put False if you do not want to sample, otherwise specify the number (int) of random samples you want to take from your dataset
sparse = True  # if False then saves all the pre-processed images from step M1
debug = False  # save debug files from extract pigmentation
quality_thresh = (
    "good"  # options are "good", "usable", "all" to use for thresholding the quality
)
