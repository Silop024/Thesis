# My own files
import parsing
import preprocessing
import classification
import feature_extraction

# Python standard libraries
import os
import sys
import argparse
import configparser

# Installed with pip
import bfi
import cv2
import joblib


def main(args: dict[str, any]):
    # Read configs
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    models = config['Models']
    test_dir = config['Paths']['test_dir']
    
    # Load classifier and pca
    clf = joblib.load(os.path.join(model_dir, models['clf']))
    pca = joblib.load(os.path.join(model_dir, models['pca']))
    
    # Load scaler and encoder
    scaler = joblib.load(os.path.join(model_dir, models['scaler']))
    encoder = joblib.load(os.path.join(model_dir, models['encoder']))
    
    # Read args
    image_path = args['image']
    correct_code = args['correctcode']
    is_verbose = args['verbose']
    input = args['input']
    correct_output = args['correctoutput']
    
    if image_path == 'HelloWorld.JPG':
        image_path = os.path.join(test_dir, image_path)
        
    # Classify shapes
    predictions = classification.classify(
        image_path=image_path, clf=clf, pca=pca, 
        scaler=scaler, encoder=encoder, debug=is_verbose)
    
    # Parse shapes into code
    code, wrong = parsing.parse(predictions, correct_code)
    
    print("Predictions")
    print(predictions)
    
    print(f'Parsed code: {code}')
    
    if len(wrong) != 0:
        image = preprocessing.read_image(image_path)
        image = preprocessing.preprocess_image(image)
        contours = feature_extraction.get_contours(image)
        debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        for i, _, parsed in wrong:
            cv2.drawContours(debug_image, contours, i, (0, 0, 255), 1)
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.putText(debug_image, parsed, (x + w // 2, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    
        cv2.imshow("Parsed", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Interpret code
    bfi.interpret(program=code, input_data=input)
    
    if correct_output is not None:
        print(f'Expected output: {correct_output}')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='app.py',
        description='Converts an image containing certain defined shapes into code'
    )
    parser.add_argument('-i', '--image', default='HelloWorld.JPG', 
                    help='the path to the image you wish to convert (default: %(default)s)'
    )
    parser.add_argument('-in', '--input', default=None,
                        help='input to give the parsed code (default: %(default)s)'
    )
    parser.add_argument('-cc', '--correctcode', default=None,
                    help='the correct code that the image represents (default: %(default)s)'
    )
    parser.add_argument('-co', '--correctoutput', default=None,
                        help='the correct output of the parsed code (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='set verbose output, will open graphs in browser (default: %(default)s)')
    
    args = vars(parser.parse_args(sys.argv[1:]))
    main(args)
    