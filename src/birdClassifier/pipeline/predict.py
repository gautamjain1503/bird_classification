import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        result=result[0].argmax()
        classes=['Asian Green Bee-Eater',
                 'Brown-Headed Barbet',
                 'Cattle Egret',
                 'Common Kingfisher',
                 'Common Myna',
                 'Common Rosefinch',
                 'Common Tailorbird',
                 'Coppersmith Barbet',
                 'Forest Wagtail',
                 'Gray Wagtail',
                 'Hoopoe',
                 'House Crow',
                 'Indian Grey Hornbill',
                 'Indian Peacock',
                 'Indian Pitta',
                 'Indian Roller',
                 'Jungle Babbler',
                 'Northern Lapwing',
                 'Red-Wattled Lapwing',
                 'Ruddy Shelduck',
                 'Rufous Treepie',
                 'Sarus Crane',
                 'White Wagtail',
                 'White-Breasted Kingfisher',
                 'White-Breasted Waterhen']
                 
        try :
            prediction = classes[result]
            return prediction
        except:
            prediction = 'Cannot predict'
            return prediction