#!/bin/bash

# https://stackoverflow.com/questions/61915607/commandnotfounderror-your-shell-has-not-been-properly-configured-to-use-conda
source /home/mauro-gwerder/anaconda3/etc/profile.d/conda.sh
conda activate PhD_new
path="/media/mauro-gwerder/Lunaphore/B17/"
mode="WSI"

#pip install --upgrade numpy

if [ "$mode" = "TMA" ]; then
  python TMA_spot_extraction.py $path
elif [ "$mode" = "WSI" ]; then
  python ROI_extraction.py $path
else
  echo "mode $mode is not a valid choice. Valid mode choices are 'WSI' or 'TMA'."
fi

#pip install numpy==1.21.6
python PhenoExtracter.py $path
python EpiSegmentor.py $path
conda activate scimap
python CellTyper.py $path
conda activate PhD_new
python TumorBudifizer.py $path
