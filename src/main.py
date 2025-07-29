from trainer import Trainer
from model import ResNet
import model_config

if __name__ == "__main__":
    trainer = Trainer(
        model_cls=ResNet,
        device=model_config.DEVICE,
        batch_size=model_config.BATCH_SIZE,
        epochs=model_config.EPOCHS,
        save_dir="../log",
        lr=model_config.LR,
    )

    # train 
    # trainer.train()

    # evaluate
    # checkpoint_path = "../log/resnet_epoch_100.pkl"
    # trainer.evaluate(checkpoint_path=checkpoint_path)

    # predict
    checkpoint_path = "../log/resnet_epoch_100.pkl"
    sample_submission_path = "../dataset/sample_submission.csv"
    submission_path = "../submission/final_submission.csv"
    trainer.predict(sample_submission_path=sample_submission_path, submission_path=submission_path, checkpoint_path=checkpoint_path)