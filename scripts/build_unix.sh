# move to root
cd ..

# prepare build folder
rm -rf build
mkdir build
cd build

# build with cmake
cmake ..
cmake --build .

# move executable to bin
mkdir bin
cp c_neural_net bin
rm c_neural_net

# remove unnecessary files
rm -rf CMakeFiles
rm cmake_install.cmake
rm CMakeCache.txt
rm Makefile

# remove unzipped data csvs (if present) to conserve space
rm data/mnist_test.csv
rm data/mnist_train.csv

# copy scripts
cp ../scripts/prepare_data_unix.sh .
cp ../scripts/run_unix.sh .


# prepare out folder
cd ..
rm -rf out
mkdir out

zip -svr out/c_neural_net_unix.zip build/ -x "*.DS_Store"
