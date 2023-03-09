import M0_Preprocess.EyeQ_process_main as M0_EQ
import M1_Retinal_Image_quality_EyePACS.test_outside as M1_EP
import M1_Retinal_Image_quality_EyePACS.merge_quality_assessment as M1_QA
import M2_Vessel_seg.test_outside_integrated as M2_VS
import M2_lwnet_disc_cup.generate_av_results as M2_DC
import extract_pigmentation
import config
import logging
import os


logging.basicConfig(level=logging.INFO,filename='data.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

   
if __name__ == "__main__":

    print("Running EyeQ process")
    # preprocessing
    M0_EQ.EyeQ_process(config)

    # Eye Quality deep learning assesment
    print("Running Image quality assesment")
    M1_EP.M1_image_quality(config)
    M1_QA.quality_assessment(config)

    # deep learning disc and vessel segmentation
    M2_VS.M2_vessel_seg(config)
    M2_DC.M2_disc_cup(config)

    # extract retinal pigmentation score
    print("extracting pigmentation")
    extract_pigmentation.get_pigmentation(config)
