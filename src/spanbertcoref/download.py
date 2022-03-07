import os
import sys
import pyhocon


def main():
    if len(sys.argv) != 2:
        print("Please specify the model name.")
        return
        
    model_name = sys.argv[1]
    config = pyhocon.ConfigFactory.parse_file(os.path.expanduser("~/.spanbertcoref.conf"))[model_name]

    maybe_download(config, model_name)



def maybe_download(config, model_name):
    if os.path.exists(os.path.join(config["model_root"], model_name, "vocab.txt")):
        print("The model is already downloaded.")
        return

    os.system(f"wget -P {config['model_root']} http://nlp.cs.washington.edu/pair2vec/{model_name}.tar.gz")
    os.system(f"tar xvzf {config['model_root']}/{model_name}.tar.gz -C {config['model_root']}")


main()