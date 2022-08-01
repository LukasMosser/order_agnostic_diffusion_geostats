from oadg.dataset import Channels


def test_channels_dataset():
    dataset = Channels()
    assert len(dataset) > 0
