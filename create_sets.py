from splitfolders import ratio
from argparse import ArgumentParser

# Original directory structure -> dataset/<Class0>, dataset/<Class1> etc.
# Finished directory structure -> train, test, validation directories with a folder
# for each class inside.

if __name__ == "__main__":
    parser = ArgumentParser(description = "Partitions dataset.")
    parser.add_argument("path", metavar = "P", type = str, help = "path to images")
    parser.add_argument("train", metavar = "T", type = float, help = "proportion for training")
    parser.add_argument("val", metavar = "V", type = float, help = "proportion for validation")
    args = parser.parse_args()

    proportions = args.train + args.val
    if proportions > 1.0:
        print("Sum of the training and validation proportions should not pass 1")

    else:

        if proportions == 1.0:
            print("Test set will be empty")
        ratio(args.path, output=".", seed = 1337, ratio = (args.train, args.val, 1.0 - args.train - args.val), group_prefix = None)


