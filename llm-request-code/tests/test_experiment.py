from req_smell_tracing.experiment import Generators, _dict_remove_none


class TestExperiment:
    def test_dict_remove_none(self):
        test_dict = {
            "a": 1,
            "b": None,
            "c": {
                "x": 2,
                "y": None,
                "z": {
                    "w": 3,
                    "v": None,
                },
            },
            "d": None,
        }

        expected_dict = {
            "a": 1,
            "c": {
                "x": 2,
                "z": {
                    "w": 3,
                },
            },
        }

        assert _dict_remove_none(test_dict) == expected_dict

    def test_generator_no_smells(self):
        reqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        reqs_smelly = [5, 7, 10, 11, 12, 13, 14, 15, 18, 19, 20]

        gen = Generators.no_smells(reqs, reqs_smelly, variations=30)

        for non_smelly, smelly in gen:
            assert len(smelly) == 0
            assert len(non_smelly) == len(reqs)

    def test_generator_all_smells(self):
        reqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        reqs_smelly = [5, 7, 10, 11, 12, 13, 14, 15, 18, 19, 20]

        gen = Generators.all_smells(reqs, reqs_smelly, variations=30)

        for non_smelly, smelly in gen:
            assert len(smelly) == len(reqs_smelly)
            assert len(non_smelly) + len(smelly) == len(reqs)
            assert len(set(non_smelly) & set(smelly)) == 0

    def test_generator_random_smells_percentage(self):
        reqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        reqs_smelly = [5, 7, 10, 11, 12, 13, 14, 15, 18, 19, 20]

        for percentage in [x / 10.0 for x in range(1, 10)]:
            gen = Generators.random_smells_percentage(
                reqs, reqs_smelly, percentage=percentage, variations=30
            )

            for non_smelly, smelly in gen:
                assert len(smelly) == int(len(reqs_smelly) * percentage)
                assert len(non_smelly) + len(smelly) == len(reqs)
                assert len(set(non_smelly) & set(smelly)) == 0

    def test_generator_random_smells(self):
        reqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        reqs_smelly = [5, 7, 10, 11, 12, 13, 14, 15, 18, 19, 20]

        gen = Generators.random_smells(reqs, reqs_smelly, variations=1000)

        samples = []
        equal_length = True
        last_length = 0

        for non_smelly, smelly in gen:
            assert len(non_smelly) + len(smelly) == len(reqs)
            assert len(set(non_smelly) & set(smelly)) == 0
            assert (non_smelly, smelly) not in samples

            samples.append((non_smelly, smelly))

            if last_length != len(smelly):
                equal_length = False

            last_length = len(smelly)

        assert not equal_length
