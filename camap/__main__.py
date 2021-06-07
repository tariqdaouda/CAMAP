import camap.trainer as MD

import pickle
import gzip
import json
import numpy
import os
import time
import argparse, glob, sys, uuid


# Note that these values can be overriden by the -ds parameter

DATASET_FOLDER = os.path.normpath(os.path.expanduser('~/.CAMAP/training_datasets'))

DATASETS = {
    243: "OneTrue_maxBs-1250_minHLANb-1_minLogFPKM-0.0003_padding-243_multiTrans-True_nonSourceFactor-5_addFields-_multiGenes-True_sequenceAlteration-removed_sourceSize-17447",
    162: "OneTrue_maxBs-1250_minHLANb-1_minLogFPKM-0.0003_padding-162_multiTrans-True_nonSourceFactor-5_addFields-_multiGenes-True_sequenceAlteration-removed_sourceSize-19658",
    81: "OneTrue_maxBs-1250_minHLANb-1_minLogFPKM-0.0003_padding-81_multiTrans-True_nonSourceFactor-5_addFields-_multiGenes-True_sequenceAlteration-removed_sourceSize-21700",
    27: "OneTrue_maxBs-1250_minHLANb-1_minLogFPKM-0.0003_padding-27_multiTrans-True_nonSourceFactor-5_addFields-_multiGenes-True_sequenceAlteration-removed_sourceSize-22084",
    9: "OneTrue_maxBs-1250_minHLANb-1_minLogFPKM-0.0003_padding-9_multiTrans-True_nonSourceFactor-5_addFields-_multiGenes-True_sequenceAlteration-removed_sourceSize-21791"
}


def train(dataset, input_size, epochs, lr, optimizer, mini_batch_size, device, mask, nb_data_workers):
    model = MD.Classifier(input_size)
    model.to(device)
    trainer = MD.Trainer(model, lr=lr, device=device, optimizer=optimizer)
    try :
        trainer.train(epochs, dataset["train"], dataset["test"], dataset["validation"], mask=mask,
                      nb_data_workers=nb_data_workers)
    except KeyboardInterrupt :
        pass

    return model, trainer


def run(job_id, padding, exp_dir, epochs, lr, optimizer, mini_batch_size,
         nb_data_workers, device, mask_values, dataset_fn):
    def _make_folder(exp_dir, job_id):
        job_base_dir = os.path.join(exp_dir, job_id)
        i = 0
        no_folder = True
        while no_folder :
            i += 1
            job_dir = "%s-%s" % (job_base_dir, i)
            try:
                os.makedirs(job_dir)
                no_folder = False
            except FileExistsError:
                pass
        os.chdir(job_dir)

    def _get_side_mask(padding, typ) :
        codon_padding = (padding//3)
        mask = numpy.ones(codon_padding * 2)
        if typ == "before":
            mask[0:codon_padding] = 0
        elif typ == "after":
            mask[codon_padding:] = 0
        elif typ == "none":
            return mask
        else:
            raise ValueError("Unknown type: %s" %typ)
        return mask

    time.sleep(5)

    mask = 1
    if "maskBefore" in mask_values:
        mask *= _get_side_mask(padding, "before")
    if "maskAfter" in mask_values:
        mask *= _get_side_mask(padding, "after")

    if type(mask) == type(0) :
        mask = None
    else:
        if sum(mask) == 0 :
            print("WARNING!!! all inputs are masked!")

    print("loading dataset...")
    with gzip.open(dataset_fn, "rb") as f:
        dataset = pickle.load(f, fix_imports=True)

    print('Job ID : #%s' % job_id)
    _make_folder(exp_dir, job_id)

    print("training...")
    input_size = int(padding // 3 * 2)
    model, trainer = train(
        dataset=dataset,
        input_size=input_size,
        epochs=epochs,
        lr=lr,
        optimizer=optimizer,
        mini_batch_size=mini_batch_size,
        device=device,
        mask=mask,
        nb_data_workers=nb_data_workers)

    print("loading best validation...")
    trainer.load_best_model("validation")
    accuracy = {}
    for setname in ["train", "validation", "test"] :
        accuracy[setname] = trainer.predict(dataset[setname], mask=mask,
                                            nb_data_workers=nb_data_workers)["accuracy"]
    print("done.")

    with open("accuracy-on-best-validation.json", "w") as f:
        json.dump(accuracy, f)

    print("Best validation accuracy:", accuracy)
    os.chdir("../..")
    return {"accuracy" : accuracy}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("padding", help="defines the dataset", type=int, action="store")
    parser.add_argument("-ds", "--dataset", help="specify custom user-generated dataset", type=str, default="")
    parser.add_argument("-e", "--epochs", type=int, default=4000, action="store")
    parser.add_argument("-lr", type=float, default=0.001, action="store")
    parser.add_argument("-mb", "--mini_batch_size", type=int, default=32, action="store")
    parser.add_argument("-n", "--experimentName", help="experiment name", type=str, default="")
    parser.add_argument("-w", "--data-workers", type=int, default=0,
                        help="number of parallel data workers")
    parser.add_argument("-o", "--optimizer", help="adagrad, adam, sgd", type=str, default="adam")
    parser.add_argument("--device", help="cpu, cuda etc...", type=str, default="cpu")
    parser.add_argument("-mbef", "--maskBefore", action="store_true",
                        help="mask sequence before peptide")
    parser.add_argument("-maft", "--maskAfter", action="store_true",
                        help="mask sequence after peptide")
    parser.add_argument("-subf", "--subfolder", help="where to store", type=str, default="")
    parser.add_argument("-cse", "--CodonShuffleEmbeddings",
                        help="Use Condon Shuffle Embeddings encoding", action="store_true")
    parser.add_argument("-aae", "--AminoAcidEmbeddings",
                        help="Use Amino Acid Embeddings encoding", action="store_true")

    args=parser.parse_args().__dict__

    # datasetShortName = "padd-%s" % args["padding"]
    if args["CodonShuffleEmbeddings"]:
        suffix = "_encoding-CodonShuffleEmbeddings.pkl.gz"
    elif args["AminoAcidEmbeddings"]:
        suffix = "_encoding-AminoAcidEmbeddings.pkl.gz"
    else:
        suffix = "_encoding-CodonEmbeddings.pkl.gz"

    padding = args["padding"]

    if args["dataset"]:
        dataset_fn = args["dataset"] + suffix
    else:
        dataset_fn = os.path.join(DATASET_FOLDER, DATASETS[padding] + suffix)

    if len(args["subfolder"]) > 0 :
        exp_dir = "./output/" + args["subfolder"]
    else :
        exp_dir = "./output"

    partial = ''
    for f in exp_dir.split('/'):
        partial += f + '/'
    try :
        os.mkdir(partial)
    except :
        pass

    nbFolders = len(glob.glob("%s/*" %exp_dir))
    if args["experimentName"] :
        job_id=args["experimentName"]
    else :
        hach = uuid.uuid4()
        job_id = "%s%s" %(hach, nbFolders)

    run(
        job_id = job_id,
        padding = padding,
        exp_dir = exp_dir,
        epochs = args["epochs"],
        lr = args["lr"],
        optimizer = args["optimizer"],
        mini_batch_size = args["mini_batch_size"],
        nb_data_workers = args["data_workers"],
        device = args["device"],
        mask_values = set([x for x in ["maskBefore", "maskAfter"] if args[x]]),
        dataset_fn = dataset_fn
    )


if __name__ == "__main__" :
    main()
