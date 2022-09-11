# LDNN-mnist


Install  
clone or unzip the archive of this repository  
  
cd LDNN-mnist  
mkdir data  
mkdir batch  
mkdir wi  
  
  
Download the dataset  
THE MNIST DATABASE of handwritten digits  
http://yann.lecun.com/exdb/mnist/  
  
put these 4 files to batch  
- t10k-images-idx3-ubyte  
- t10k-labels-idx1-ubyte  	
- train-images-idx3-ubyte  	
- train-labels-idx1-ubyte  
  
  
Prepare  
1. check path fot LesserDNN  
in .py  
  
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))  
  
this line includes LesserDNN framework:)  
check if "../ldnn" is correct on your env.  
  
2.  
python3 ./make_batch  
  
this will create batch files for training and test  
  
3.  
./test.sh  
  
  
Training  
./train.sh  
  
  
Changeing platform  
modiy "../ldnn/plat.py" to change platfoms(=opencl or cupy)   



