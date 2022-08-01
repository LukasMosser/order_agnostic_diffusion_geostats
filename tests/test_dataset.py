from oadg.dataset import Channels


def test_channels_dataset():
    dataset = Channels(root=".", download=True)
    assert len(dataset) > 0
