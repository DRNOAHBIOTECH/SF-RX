#!/usr/bin/env python3
"""
Train a ECN-Transformer model on the DDI dataset
"""
# %%
from my_utils.config import get_args_Transformer

def main():
    args = get_args_Transformer()

    import os

    from lightning.pytorch.loggers import CSVLogger
    import lightning as L 
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.callbacks import ModelCheckpoint

    from my_utils.config import FilePaths 
    from my_utils.prep_ddi import DDIDataProcessor
    from my_utils.GNN_data_set_loader import DDI_GCN_Dataset
    from my_utils.GNN_data_set_loader import GNN_DataSplitterByLevel  
    from models.GNN import generate_gnn_output_config
    from models.GNN import FlexibleLitGNNModel
    from models.Transfomer import ECNTransformer 

    file_paths = FilePaths()
    file_paths.check_paths_exist()
    ddi_processor = DDIDataProcessor(
        file_paths = file_paths.as_dict())
    ddi_processor.process()

    DDI_GCN_Dataset.set_ddi_info(ddi_processor)
    splitter = GNN_DataSplitterByLevel(
        ddi_processor.ddi_label_df,
        DDI_GCN_Dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers_for_dt,)
    # A list of dictionaries containing the train and val DataLoaders
    fold_dl = splitter.split_data(method=args.split_level)  # Choose method 1, 2, 3, or 4

    batch = next(iter(fold_dl[0]['val']))
    NUM_NODE_FEATURES = batch.x.shape[1]
    NUM_EDGE_FEATURES = batch.edge_attr.shape[1]
    LOG_PATH = f"ECN_Transformer_LOG/logs_{args.split_level}_split"

    CLASS_CONFIGS = [
        {'name': 'severity', 
        'num_classes': ddi_processor.NUM_SEVERITY_CLASSES, 
        'classification_type': args.classification_type,
        },

        {'name': 'desc', 
        'num_classes': ddi_processor.NUM_DESC_CLASSES,
        'classification_type': args.classification_type,
        },
    ]
    output_configs = generate_gnn_output_config(*CLASS_CONFIGS)

    if isinstance(args.folds, int):
        FOLDS = range(args.folds)
    else:
        FOLDS = args.folds

    for fold_idx in FOLDS:
        model = ECNTransformer(
            num_node_features=NUM_NODE_FEATURES,
            num_edge_features=NUM_EDGE_FEATURES, 
            output_configs = output_configs,
            dropout_rate=args.dropout_rate,
            expand_factor=args.expand_factor,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,)

        lit_model = FlexibleLitGNNModel(
            model, output_configs = output_configs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size)
        logger = CSVLogger(LOG_PATH)
        train_loader = fold_dl[fold_idx]['train']
        val_loader = fold_dl[fold_idx]['val']
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{LOG_PATH}/checkpoints",
            # filename: Base filename for the checkpoint, the filename is completed 
            # with the epoch and the value of the monitored quantity
            filename=f"fold_{fold_idx}_best_model",
            monitor='val_f1_macro_desc',
            mode='max',
            save_top_k=1, # Save only the best model
            verbose=True
        )
        callbacks = [checkpoint_callback]
        if args.enable_early_stopping:
            # Early stopping
            early_stop_callback = EarlyStopping(
                monitor='val_f1_macro_desc',
                patience=10,
                verbose=False,
                mode='max'
            )
            callbacks.append(early_stop_callback)
            trainer = L.Trainer(
                devices= [args.device_num],
                max_epochs=args.epochs,
                logger=logger,
                callbacks=callbacks)
        else:
            trainer = L.Trainer(
                devices= [args.device_num],
                max_epochs=args.epochs,
                logger=logger)

        if fold_idx == 0:
            hyperparameters = dict(
                num_node_features=NUM_NODE_FEATURES,
                num_edge_features=NUM_EDGE_FEATURES,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_split=f'{args.split_level}_split',
                log_path=LOG_PATH,
                script_name = os.path.realpath(__file__),
                weight_decay = args.weight_decay,
                dropout_rate = args.dropout_rate,
                output_configs = output_configs,
                classification_type = args.classification_type,
                expand_factor = args.expand_factor,
                nhead = args.nhead,
                num_encoder_layers = args.num_encoder_layers) 
            trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(lit_model, train_loader, val_loader)

if __name__ == '__main__':
    main()