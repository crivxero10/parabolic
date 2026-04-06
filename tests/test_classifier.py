import unittest

from parabolic.classifier import Parity, Regime, RegimeClassifier, RegimeClassifierConfig


class TestClassifier(unittest.TestCase):
    def test_classifier_get_regime__trivial_bull(self):
        config = RegimeClassifierConfig(
            k_st=6,
            k_lt=42,
            lookback=11,
            crab_lower_bound=-2.0,
            crab_upper_bound=2.0,
        )
        classifier = RegimeClassifier(config)

        classifier.state.blue_curve = [10.0] * config.lookback
        classifier.state.green_curve = [0.0] * config.lookback
        classifier.state.momentum = 0
        classifier.state.current_parity = Parity.DOJI

        regime = classifier.get_regime(classifier.state, classifier.config)

        assert regime == Regime.BULL
    
    def test_classifier_get_regime__trivial_bear(self):

        config = RegimeClassifierConfig(
            k_st=6,
            k_lt=42,
            lookback=11,
            crab_lower_bound=-2.0,
            crab_upper_bound=2.0,
        )
        classifier = RegimeClassifier(config)

        classifier.state.blue_curve = [0.0] * config.lookback
        classifier.state.green_curve = [10.0] * config.lookback
        classifier.state.momentum = 0
        classifier.state.current_parity = Parity.DOJI

        regime = classifier.get_regime(classifier.state, classifier.config)

        assert regime == Regime.BEAR

    def test_classifier_get_regime__trivial_crab(self):
        config = RegimeClassifierConfig(
            k_st=6,
            k_lt=42,
            lookback=11,
            crab_lower_bound=-2.0,
            crab_upper_bound=2.0,
        )
        classifier = RegimeClassifier(config)

        classifier.state.blue_curve = [
            0.1,
            -0.1,
            0.1,
            -0.1,
            0.1,
            -0.1,
            0.1,
            -0.1,
            0.1,
            -0.1,
            0.1,
        ]
        classifier.state.green_curve = [0.0] * config.lookback
        classifier.state.momentum = 0
        classifier.state.current_parity = Parity.DOJI

        regime = classifier.get_regime(classifier.state, classifier.config)

        assert regime == Regime.CRAB

    def test_classifier_get_regime__transition_crab_to_bull_easy_case(self):
        config = RegimeClassifierConfig(
            k_st=6,
            k_lt=42,
            lookback=11,
            crab_lower_bound=-2.0,
            crab_upper_bound=2.0,
        )
        classifier = RegimeClassifier(config)

        classifier.state.blue_curve = [
            0.0,
            0.25,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ]
        classifier.state.green_curve = [0.0] * config.lookback
        classifier.state.momentum = 0
        classifier.state.current_parity = Parity.DOJI

        regime = classifier.get_regime(classifier.state, classifier.config)

        assert regime == Regime.BULL

    def test_classifier_get_regime__transition_bear_to_crab_easy_case(self):
        config = RegimeClassifierConfig(
            k_st=6,
            k_lt=42,
            lookback=11,
            crab_lower_bound=-20.0,
            crab_upper_bound=20.0,
        )
        classifier = RegimeClassifier(config)

        classifier.state.blue_curve = [
            -2.5,
            -1.5,
            -0.75,
            -0.25,
            0.0,
            0.1,
            -0.1,
            0.1,
            -0.1,
            0.1,
            -0.1,
        ]
        classifier.state.green_curve = [1.0] * config.lookback
        classifier.state.momentum = 0
        classifier.state.current_parity = Parity.DOJI

        regime = classifier.get_regime(classifier.state, classifier.config)

        assert regime == Regime.CRAB
