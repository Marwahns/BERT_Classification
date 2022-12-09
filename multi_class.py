from utils.preprocessor_class import PreprocessorClass
from models.multi_class_model import MultiClassModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    #"bert_classification/data/preprocessed"
    dm = PreprocessorClass(preprocessed_dir = "data/preprocessed",
                 batch_size = 10,
                 max_length = 100)
                 
    model = MultiClassModel(
        n_out = 5,
        # dropout tentuin sendiri
        dropout = 0.3,    

        # 1e-3 = 0.0001   
        lr = 1e-5
    )

    logger = TensorBoardLogger("logs", name="bert-multi-class")

    trainer = pl.Trainer(
        gpus = 1,
        max_epochs = 10,
        #"bert_classification/checkpoints/class"
        default_root_dir = "./checkpoints/class"
    )

    trainer.fit(model, datamodule = dm)
    pred, true = trainer.predict(model = model, datamodule = dm)