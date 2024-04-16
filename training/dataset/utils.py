def get_dataset(args, **kwargs):
    """Function to get the specific dataset

    Args:
        args (argparse.Namespace): Arguments from the command line.

    Returns:
        Dataset: The dataset object.
    """    
    if args.dimension == "2d":
        assert args.dimension == "2d" "2D dataset is not implemented yet"
        pass

    else:
        if args.dataset == "btcv":
            from .dim3.dataset_btcv import BTCVDataset
            print(type(BTCVDataset(args)))
            return BTCVDataset(args)