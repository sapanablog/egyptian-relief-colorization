###I have done for this for imagenet testing eval.sh
#PRED_DIR='/home/sapanagupta/PycharmProjects/color2/iColoriT/predictions'
#GT_DIR='/home/sapanagupta/PycharmProjects/color2/iColoriT/Doc2/imgs'
#NUM_HINT=${2:-100}

#this is done in a small 150 train, val 50,test 50 for egyptian
#PRED_DIR='/home/sapanagupta/PycharmProjects/color2/iColoriT/output_dir/testing_data/predictions'
#GT_DIR='/home/sapanagupta/PycharmProjects/color2/iColoriT/output_dir/testing_data/imgs/'
#NUM_HINT=${2:-100}

##this is done by 462 train, test -99,val-99 scratch model for egyptian
#PRED_DIR='/home/sapanagupta/PycharmProjects/color2/iColoriT/data/train_data/output_dir/test_data/predictions'
#GT_DIR='/home/sapanagupta/PycharmProjects/color2/iColoriT/data/train_data/output_dir/test_data/imgs'
#NUM_HINT=${2:-100}

##this is done by 462 train, test -99,val-99 scratch evaluating for each hints
#PRED_DIR='/media/sapanagupta/vol1/Sapana/data/train_data/output_dir_old/test_data/predictions'
#GT_DIR='/media/sapanagupta/vol1/Sapana/data/train_data/output_dir_old/test_data/imgs'
##NUM_HINT=${0:-100}
## Array of hint counts to evaluate
#HINT_COUNTS=(0 1 2 5 10 20 50 100 200)
#
#for NUM_HINT in "${HINT_COUNTS[@]}"
#do
#    echo "Evaluating for ${NUM_HINT} hints"
#
#
## other options
#opt=${3:-}
#
## batch_size can be adjusted according to the graphics card
#python /home/sapanagupta/PycharmProjects/color2/iColoriT/evaluation/evaluate.py \
#    --pred_dir=${PRED_DIR} \
#    --gt_dir=${GT_DIR} \
#    --num_hint=${NUM_HINT} \
#
#    $opt
#
#done

##########This is done for scratch#################
##########              #################
###########################
##this is done by 462 train, test -99,val-99 scratch evaluating for each hints
#PRED_DIR='/media/sapanagupta/vol1/Sapana/data/train_data/output_dir_old/test_data/predictions'
#GT_DIR='/media/sapanagupta/vol1/Sapana/data/train_data/output_dir_old/test_data/imgs'
#RESULTS_FILE="psnr_results.txt"
#
## Clear the results file if it exists
#> $RESULTS_FILE
#
## Array of hint counts to evaluate
#HINT_COUNTS=(0 1 2 5 10 20 50 100 200)
#
## other options
#opt=${3:-}
#
#for NUM_HINT in "${HINT_COUNTS[@]}"
#do
#    echo "Evaluating for ${NUM_HINT} hints"
#    RESULT=$(python /home/sapanagupta/PycharmProjects/color2/iColoriT/evaluation/evaluate.py \
#        --pred_dir=${PRED_DIR} \
#        --gt_dir=${GT_DIR} \
#        --num_hint=${NUM_HINT})
#
#         $opt
#
#    # Extract PSNR value from the result
#    PSNR=$(echo "$RESULT" | grep -oP 'PSNR: \K[0-9.]+')
#
#    # Save the result to the file
#    echo "Hints: ${NUM_HINT}, PSNR: ${PSNR}" >> $RESULTS_FILE
#
#    echo "Hints: ${NUM_HINT}, PSNR: ${PSNR}"
#done
#
#echo "Results saved in $RESULTS_FILE"
#
#
###########################
########## complete #################
##########    complete          #################
#
#
##########This is done for pretrained model of icolorit 4base model on imagenet test dataset to calculate hints for respective psnr################
##########              #################
###########################

##this is done for icolorit pretrained model of test dataset imagenet... evaluating for each hints#####
#PRED_DIR='/media/sapanagupta/vol1/Sapana/predictions'
#GT_DIR='/home/sapanagupta/PycharmProjects/color2/iColoriT/Doc2/imgs'
#RESULTS_FILE="psnr_results.txt"
#
## Clear the results file if it exists
#> $RESULTS_FILE
#
## Array of hint counts to evaluate
#HINT_COUNTS=(0 1 2 5 10 20 50 100 200)
#
## other options
#opt=${3:-}
#
#for NUM_HINT in "${HINT_COUNTS[@]}"
#do
#    echo "Evaluating for ${NUM_HINT} hints"
#    RESULT=$(python /home/sapanagupta/PycharmProjects/color2/iColoriT/evaluation/evaluate.py \
#        --pred_dir=${PRED_DIR} \
#        --gt_dir=${GT_DIR} \
#        --num_hint=${NUM_HINT})
#
#         $opt
#
#    # Extract PSNR value from the result
#    PSNR=$(echo "$RESULT" | grep -oP 'PSNR: \K[0-9.]+')
#
#    # Save the result to the file
#    echo "Hints: ${NUM_HINT}, PSNR: ${PSNR}" >> $RESULTS_FILE
#
#    echo "Hints: ${NUM_HINT}, PSNR: ${PSNR}"
#done
#
#echo "Results saved in $RESULTS_FILE"
#
#
##########   complete           #################
##########     complete         #################
##########   complete           #################


##########This is done for pretrained model of icolorit 4base model on imagenet test dataset to calculate hints for respective psnr################
##########              #################
###########################

##this is done for finetune model of test dataset egyptian ... evaluating for each hints#####
#PRED_DIR='/media/sapanagupta/vol1/Sapana/data/test_data/predictions'
#GT_DIR='/media/sapanagupta/vol1/Sapana/data/test_data/imgs'
#RESULTS_FILE="psnr_results.txt"


################################
##this is done for finetune model of test dataset egyptian ... evaluating for each hints#####
#PRED_DIR='/media/sapanagupta/vol1/Sapana/data/Threshold/Test_threshold/pred'
#GT_DIR='/media/sapanagupta/vol1/Sapana/data/Threshold/Test_threshold/imgs'
#RESULTS_FILE="/media/sapanagupta/vol1/Sapana/data/Threshold/Test_threshold/psnr_results.txt"


################################
##this is done for finetune model of test dataset egyptian ... evaluating for each hints#####
#PRED_DIR='/media/sapanagupta/vol1/Sapana/data/Threshold/predictions'
#GT_DIR='/media/sapanagupta/vol1/Sapana/data/test_data/imgs'
#RESULTS_FILE="/media/sapanagupta/vol1/Sapana/data/Threshold/Test_threshold/psnr_results.txt"

#/media/sapanagupta/vol1/Sapana/data/Threshold/predictions


##############################
#this is done for finetune model of test dataset egyptian ... evaluating for each hints#####
#PRED_DIR='/media/sapanagupta/vol1/Sapana/data/Threshold/Test_threshold/Test_Color_pred/prediction/'
#GT_DIR='/media/sapanagupta/vol1/Sapana/data/Threshold/Test_threshold/imgs'
#RESULTS_FILE="/media/sapanagupta/vol1/Sapana/data/Threshold/Test_threshold/psnr_results.txt"

######sam2 generated mask for hints generation...
#PRED_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/SAM2_output_dir/prediction/'
#GT_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/SAM2_gen_labels/TEST/imgs'
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/SAM2_gen_labels/psnr_results.txt"

#############sam2 generated mask with threshold 90 for 20 hints
#
#PRED_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/SAM2_output_dir_90/prediction_90/'
#GT_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/SAM2_gen_labels_90/TEST/imgs'
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/SAM2_gen_labels/psnr_results.txt"
#

###sam2 generated mask with thresh = 90, for 200 hints
#PRED_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/SAM2_output_dir_90/200_hints/prediction_200/'
#GT_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/SAM2_gen_labels_90/TEST200/imgs'
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/SAM2_gen_labels/psnr_results.txt"


######################applied for optimal threshold 60 to images
#PRED_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/appl_60_thres/test_appl_60_thres/prediction_60/'
#GT_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/appl_60_thres/test_appl_60_thres/imgs'
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/SAM2_gen_labels/psnr_results.txt"

############this is done for resnet50 using unet , generated mask from resnet50
#PRED_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/resnet_50/test/pred_resnet_50/'
#GT_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/resnet_50/test/imgs/'
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/resnet_50/test/psnr_results.txt"

######################this is for 172 optimal threshold applied
#PRED_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/test/thres_172/pred_172_opt/'
#GT_DIR='/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/test/thres_172/imgs/'
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/test/thres_172/psnr_results.txt"
#

####sam2finetune at 100
#PRED_DIR='/home/sapanagupta/ICOLORIT_INPUTS/OI/sam2/sam2_mask/test/pred_sam2_finetune/'
#GT_DIR='/home/sapanagupta/ICOLORIT_INPUTS/OI/sam2/sam2_mask/test/imgs/'
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/OI/sam2/sam2_mask/test/psnr_results.txt"
#

####this is for cluster and resnet50
##GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/"
#PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/pred_cluster_resnet50/"
#GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/imgs/"
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/evaluation_results1.txt"

#####this is FPS and saliency using resnet mask50 using h4size hints
####this is for cluster and resnet50
##GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/"
#PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/cluster_4size/prediction/"
#GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/cluster_4size/imgs/"
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Test/evaluation_results1.txt"

###this is for saturated region using saligency, cluster,clahe on resnet50 mask
#PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_cluster/Test/prediction/"
#GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_cluster/Test/imgs/"
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_cluster/Test/evaluation_results1.txt"

####saturated region for binary mask then filter FPS at threshold 25
###this is for saturated region using saligency, cluster,clahe on resnet50 mask
#PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/prediction/"
#GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/imgs/"
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/evaluation_results1.txt"

######this is for without lambda using fps ,saligency,
#
#PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/prediction_Fine_without_Lambda/"
#GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/imgs/"
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/evaluation_results1.txt"
#
############################This is for adaptive weight for lambada tpr
#
#PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/prediction_Fine_adapt_lambda_1/"
#GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/imgs/"
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Sat_thres/Test/evaluation_results1.txt"

########this is evaluating for the original edfu temple dataset for 15 test images
#
#
#PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Resnet50_108_retrain/Test/pred/"
#GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Resnet50_108_retrain/Test/imgs/"
#RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ML_INPUTS/Resnet50_108_retrain/Test/evaluation_results1.txt"
#

#####################this is ablation for clahe and saturation

PRED_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/output_Thesis/checkpoint-0/"
GT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/INPUT_Thesis/Sapana/july_2nd_meet_work/Tast_patch_full_res_train/ICOLORIT_INPUTS/INPUTS/data/Test/"
RESULTS_FILE="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/INPUT_Thesis/Sapana/july_2nd_meet_work/Tast_patch_full_res_train/ICOLORIT_INPUTS/INPUTS/data/Test/evaluation_results_hints_psnr.txt"

################################
# Clear the results file if it exists
> $RESULTS_FILE

# Array of hint counts to evaluate
HINT_COUNTS=(0 1 2 5 10 20 )

# other options
opt=${3:-}

for NUM_HINT in "${HINT_COUNTS[@]}"
do
    echo "Evaluating for ${NUM_HINT} hints"
    RESULT=$(python /home/sapanagupta/PycharmProjects/color2/iColoriT/evaluation/evaluate.py \
        --pred_dir=${PRED_DIR} \
        --gt_dir=${GT_DIR} \
        --num_hint=${NUM_HINT})

         $opt

    # Extract PSNR value from the result
    PSNR=$(echo "$RESULT" | grep -oP 'PSNR: \K[0-9.]+')

    # Save the result to the file
    echo "Hints: ${NUM_HINT}, PSNR: ${PSNR}" >> $RESULTS_FILE

    echo "Hints: ${NUM_HINT}, PSNR: ${PSNR}"
done

echo "Results saved in $RESULTS_FILE"


#########   complete           #################
#########     complete         #################
#########   complete           #################

