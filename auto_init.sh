cd /content/drive/My\ Drive/Colab\ Notebooks/MotionPredictionFiles
cp data_utils.py /content
cp forward_kinematics.py /content
cp seq2seq_model.py /content
cp translate.py /content
cp auto_train.sh /content
cd /content

pip uninstall tensorflow --yes
pip uninstall tensorboard --yes
yes | pip install tensorflow-gpu==1.13.1
yes | pip install matplotlib

