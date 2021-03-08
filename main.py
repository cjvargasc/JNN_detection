from train import Trainer
from test import Tester


def train():
    Trainer.train()


def test():
    Tester.test()


if __name__ == "__main__":
    train()
    test()
    #Tester.test_one_OL()
    #Tester.test_one_COCO()
