# Download metric evaluation weights
cd src/metric
mkdir -p checkpoints
gdown --folder "https://drive.google.com/drive/folders/1MVddNupLwUWUWuyGlqEDFiUT0gL1Dyuh?usp=sharing" -O checkpoints
gdown --folder "https://drive.google.com/drive/folders/1F8fy8eP3-GFpp_Udmi-BsFb40fJlas4W?usp=sharing" -O checkpoints
cd ../..

# Download SMPL model
cd data
mkdir smpl
cd smpl
gdown 11k8_VE6mA7x5LEIWE02uyMSe5ntgVSvD 
cd ../..

# Model weights
cd outputs
gdown 1SsG8iYmMU7jwcUA1pYczL96G3VIVNJIb
gdown 1oDr1f51JPM1BaCQnRE__zVP1P4QICJrI
unzip STREAM_HM.zip
unzip STREAM.zip
rm -rf STREAM_HM.zip STREAM.zip
cd ../..