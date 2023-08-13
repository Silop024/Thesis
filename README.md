# Set up
cd your-path/Thesis

python3 -m venv env

env/Scripts/activate

pip install -r requirements.txt

# Usage
cd your-path/Thesis

### If the environment is not already activated
./env/Scripts/Activate.ps1

### Instructions for how to run the program
usage: src/app.py [-h] [-i IMAGE] [-in INPUT] [-cc CORRECTCODE] [-co CORRECTOUTPUT] [-v]

Converts an image containing certain defined shapes into code, the definition of the shapes can be found in config.ini.

options:

  -h, --help            show this help message and exit
  
  -i IMAGE, --image IMAGE
                        the path to the image you wish to convert (default: HelloWorld.JPG)
                        
  -in INPUT, --input INPUT
                        input to give the parsed code (default: None)
                        
  -cc CORRECTCODE, --correctcode CORRECTCODE
                        the correct code that the image represents (default: None)
                        
  -co CORRECTOUTPUT, --correctoutput CORRECTOUTPUT
                        the correct output of the parsed code (default: None)
                        
  -v, --verbose         set verbose output, will open graphs in browser (default: False)

#### Example
(env) PS your-path\Thesis> src/app.py

Output: Hello World!