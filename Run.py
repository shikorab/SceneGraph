import sys
from Module.Train import train
from Module.Eval import eval

if __name__ == "__main__":
    application = None
    name = None
    gpu = None
    # get input parameters

    # application
    if len(sys.argv) > 1:
        application = sys.argv[1]

    # module name
    if len(sys.argv) > 2:
        name = sys.argv[2]

    # gpu number
    if len(sys.argv) > 3:
        gpu = sys.argv[3]

    if application == "train":
        train(name=name, gpu=gpu)
    elif application == "eval":
        eval(load_module_name=name, gpu=gpu)
    elif application == "download":
        print("TBD")
    else:
        # print usage
        print("Error: unexpected usage\n\n")
        print("SGP Runner")
        print("----------")
        print("Download data: \"python Run.py download\"")
        print("               Should be run just once, on the first time the module used")
        print("Train Module: \"python Run.py train <module_name> <gpu_number>\"")
        print("               Train lingustic SGP")
        print("               Module weights with the highest score over the validation set will be saved as \"<module_name>_best\"")
        print("               Module weights of the last epoch will be saved as \"<module_name>\"")
        print("Eval Module: \"python Run.py eval <module_name> <gpu_number>\"")
        print("               Recall@100 evaluation for the trained module.")
        print("               Use 'gpi_ling_orig_best' for a pre-trained module")
        print("               Use \"<module_name>_best\" for a self-trained module")
