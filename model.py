import numpy as np
import onnxruntime as ort
from PIL import Image
from resizeimage import resizeimage

class Model:
    def __init__(self, onnx_path):
        self.model_path = onnx_path
        self.session = None
        self.load_model()

    def load_model(self):
        """Load the ONNX model."""
        self.session = ort.InferenceSession(self.model_path)

    def preprocess(self, inputImage):
        """Preprocess the image for ONNX inference."""
        img = resizeimage.resize_cover(inputImage, [224,224], validate=False)
        img_ycbcr = img.convert('YCbCr')
        img_y_0, img_cb, img_cr = img_ycbcr.split()
        img_ndarray = np.asarray(img_y_0)

        img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
        img_5 = img_4.astype(np.float32) / 255.0
        return img_cb, img_cr, img_5

    def predict(self, processed_image):
        """Run model prediction on the preprocessed image."""
        # Assuming the model's input name is 'input' and output name is 'output'
        # Adjust these names based on your specific ONNX model
        ort_inputs = {self.session.get_inputs()[0].name: processed_image} 
        ort_outs = self.session.run(None, ort_inputs)
        return ort_outs[0]

    def postprocess(self, prediction, img_cb, img_cr):
        img_out_y = Image.fromarray(np.uint8((prediction[0] * 255.0).clip(0, 255)[0]), mode='L')
        final_img = Image.merge(
            "YCbCr", [
                img_out_y,
                img_cb.resize(img_out_y.size, Image.BICUBIC),
                img_cr.resize(img_out_y.size, Image.BICUBIC),
            ]).convert("RGB")
        return final_img

    def process_image(self, inputImage):
        """Entry method to process an image."""
        img_cb, img_cr, processed_image = self.preprocess(inputImage)
        predictions = self.predict(processed_image)
        print('input:', processed_image.shape, '  output:', predictions.shape)
        return self.postprocess(predictions, img_cb, img_cr)

# Example Usage:
# model = Model('path_to_model.onnx')
# result = model.process_image(input_image)