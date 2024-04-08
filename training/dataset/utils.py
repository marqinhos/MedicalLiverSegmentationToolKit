def get_dataset(args, mode, **kwargs):
    if args.dimension == "2d":
        assert args.dimension == "2d" "2D dataset is not implemented yet"
        pass

    else:
        if args.dataset == "btcv":
            from .dim3.dataset_btcv import BCTVDataset

            return BCTVDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)