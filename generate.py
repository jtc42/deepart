import numpy as np
from scipy.ndimage import imread

from fet_extractor import load_fet_extractor
from deepart import gen_target_data, optimize_img


# Calculate the size of images to use, based on the input and an optional width
def calculate_shape(image, length=0):
    im = imread(image)
    shape = [int(im.shape[0]),int(im.shape[1])]
    if length:
        shape = [int(d*(length/float(shape[1]))) for d in shape]
    return tuple(shape)

    
def setup_classifier(image_dims):

    model_path = 'C:/Toolkits/caffe/models/VGG_CNN_19/' 

    deployfile_relpath = model_path + 'VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'
    weights_relpath = model_path + 'vgg_normalised.caffemodel'

    mean = (103.939, 116.779, 123.68) #Specific to VGG19 normalised
    device_id = 0 #GPU device ID
    input_scale = 1.0

    caffe, net = load_fet_extractor(
        deployfile_relpath, weights_relpath, image_dims, mean, device_id,
        input_scale
    )

    return caffe, net


##MAIN FUNCTION##

def deepart(content, style, init_noise = False, length = 0, style_weight = 100, display = 50, max_iter = 350):
    """
    content: String - Path to content image
    style: String - Path to style image
    init_noise: Bool - If true, noise will be used as initial image instead of content
    length: Int - Length of image to process (reduce this if running into out of memory errors)
    style_weight: Int - Ratio of style to content
    display: Int - Number of iterations between saved images
    max_iter: Int - Maximum number of iterations before stopping
    """
    
    #Seed noise
    np.random.seed(123)

    #Folder to store results
    root_dir = 'results'
    
    #Save image every n=display iterations
    display = display
    max_iter = max_iter
    
    # list of targets defined by tuples of
    # (
    #     image path,
    #     target blob names (these activations will be included in the loss function),
    #     whether we use style (gram) or content loss,
    #     weighting factor
    # )
    targets = [
        (style, ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], True, style_weight),
        (content, ['conv4_2'], False, 1),
    ]
    # These have to be in the same order as in the network
    all_target_blob_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']


    # Initialize content and style targets
    image_dims = calculate_shape(content, length=length) #Calculate size of image to process
    print "Running for image dimensions:  " + str(image_dims)
    caffe, net = setup_classifier(image_dims) #Set up network based on calculated image size
    

    # Generate activations for input images
    target_data_list = gen_target_data(root_dir, caffe, net, targets)


    # Create init image
    if init_noise:
        # Generate white noise image
        init_img = np.random.normal(loc=0.5, scale=0.1, size=image_dims + (3,))
    else:
        #Get from content image and rescale
        init_img = caffe.io.load_image(targets[1][0])
        init_img = caffe.io.resize_image(init_img,image_dims)
    
    
    # Set up solver
    solver_type = 'L-BFGS-B'
    solver_param = {}


    # Start optimizing
    optimize_img(
        init_img, solver_type, solver_param, max_iter, display, root_dir, net,
        all_target_blob_names, targets, target_data_list
    )


import optparse
#deepart(content, style, init_noise = False, length = 0, style_weight = 100, display = 50, max_iter = 350):

# Run demo
if __name__ == '__main__':
    """
    style_demo = 'images/scream.jpg'
    content_demo = 'images/profile.jpg'
    deepart(content_demo, style_demo, init_noise=False, display = 50, max_iter = 150, length=600)
    """

    parser = optparse.OptionParser()
    parser.add_option('-c', '--content', 
                      dest="content_path", 
                      default="images/sanfrancisco.jpg",
                      )
    parser.add_option('-s', '--style',
                      dest="style_path",
                      default="images/starry.jpg",
                      )
    parser.add_option('-n','--noise',
                      dest="init_noise",
                      default=False,
                      )
    parser.add_option('-l','--length',
                      dest="length",
                      default=0,
                      type="int",
                      )
    parser.add_option('-r','--ratio',
                      dest="style_weight",
                      default=100,
                      type="int",
                      )
    parser.add_option('-d','--display',
                      dest="display",
                      default=50,
                      type="int",
                      )
    parser.add_option('-i','--iterations',
                      dest="max_iter",
                      default=300,
                      type="int",
                      )
    options, remainder = parser.parse_args()
    
    print 'CONTENT   :', options.content_path
    print 'STYLE   :', options.style_path
    print 'NOISE    :', options.init_noise
    print 'LENGTH :', options.length
    print 'RATIO :', options.style_weight
    print 'DISPLAY :', options.display
    print 'ITERATIONS :', options.max_iter
    
    deepart(options.content_path, options.style_path, init_noise=options.init_noise, length = options.length, style_weight = options.style_weight, display = options.display, max_iter = options.max_iter)
