# deepart 

Modification of kovibalu's implementation of neural style transfer. This versions adds the ability to use the content image as the initialisation image, giving results better matching the content, in fewer iterations. I've also implemented functionality to pass arguments through the command line, saving modification of the .py files.

##Getting started

Run `./scripts/install/install_all.sh` to install. It will install the needed python dependencies, caffe, compile caffe, download the necessary weight file. If running Windows, ensure all dependencies are installed, download the pre-trained network from [here](http://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel), and place the file in 'models/VGG_CNN_19/'.

If you already have the normalized VGG19 files stored elsewhere, edit 'generate.py' variable 'model_path' to reflect this.

Once everything is installed, run 'python generate.py' from command prompt/terminal to run a demo. Results will be saved to '/results'.

##Using the script

```
python generate.py -c <content image> -s <style image> -l <width of image> -i <n iterations>
```
If no image width is specified, the dimensions of the content image will be used, so be careful if running on limited VRAM.
Similarly, if no iteration number is specified it will default to 300. This gives good results when initialising with content, but when initialising with noise I suggest increasing this to at least 500.

###Additional arguments
```
-n <True/False>
``` 
Initialise with noise instead of content image (default False)
```
-r <ratio>
``` 
Ratio of style to content (default 100)
```
-d <display iterations>
``` 
Number of iterations between image saves

## Examples

Content initialization, 300 iterations
<p align="center">
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/images/chantry.jpg" width="40%"/>
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/results/chantry-content-300-it.jpg" width="40%"/>
</p>

Noise initialization, 300 iterations
<p align="center">
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/images/chantry.jpg" width="40%"/>
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/results/chantry-noise-300-it.jpg" width="40%"/>
</p>

Content initialization, 150 iterations
<p align="center">
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/images/sanfrancisco.jpg" width="40%"/>
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/results/sanfran-content-150-it.jpg" width="40%"/>
</p>

Noise initialization, 1000 iterations
<p align="center">
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/images/sanfrancisco.jpg" width="40%"/>
<img src="https://raw.githubusercontent.com/jtc42/deepart/master/results/sanfran-noise-1000-it.jpg" width="40%"/>
</p>
Original image: [San Francisco](https://www.flickr.com/photos/anhgemus-photography/15377047497) by Anh Dinh. All images were released under the Creative Comments license.


This is an implementation of the original neural style project by [Gatys et al.](https://arxiv.org/abs/1508.06576).
